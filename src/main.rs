use std::{convert::Infallible, time::Duration};
use std::{env, fmt::Write};

use dotenv::dotenv;
use llm::{
    conversation_inference_callback, load_progress_callback_stdout, models::Llama,
    InferenceParameters, InferenceRequest, KnownModel, ModelParameters, Prompt,
};
use once_cell::sync::{Lazy, OnceCell};
use rand::Rng;
use serenity::{
    all::{GetMessages, Message, Ready},
    async_trait,
    prelude::*,
};
use tokio::{sync::Semaphore, task::spawn_blocking, time::sleep};

//

static TASKS: Lazy<Semaphore> = Lazy::new(|| Semaphore::new(4));
static BOT: Lazy<Llama> = Lazy::new(|| {
    Llama::load(
        "./asset/Wizard-Vicuna-7B-Uncensored.ggmlv3.q8_0.bin".as_ref(),
        llm::TokenizerSource::Embedded,
        ModelParameters {
            context_size: 8096,
            // use_gpu: true,
            ..<_>::default()
        },
        load_progress_callback_stdout,
    )
    .unwrap()
});
static BOT_NAME: OnceCell<String> = OnceCell::new();
static BOT_PERSONA: OnceCell<String> = OnceCell::new();

//

fn prompt(messages: String) -> String {
    let (Some(bot_name), Some(persona)) = (BOT_NAME.get(), BOT_PERSONA.get()) else {
        return String::new();
    };

    let mut rng = rand::thread_rng();
    let mut sess = BOT.start_session(<_>::default());
    let mut buf = String::new();

    let prompt = format!(
        "{persona}\n\nContext:\n\
            {messages}\
            {bot_name}:"
    );

    println!("RUNNING `{prompt}`");
    let err = sess.infer::<Infallible>(
        &*BOT,
        &mut rng,
        &InferenceRequest {
            prompt: Prompt::Text(&prompt),
            parameters: &InferenceParameters::default(),
            play_back_previous_tokens: false,
            maximum_token_count: None,
        },
        &mut <_>::default(),
        conversation_inference_callback("###", |s| {
            // use std::io::Write;
            // print!("{s}");
            // stdout().flush().unwrap();
            buf.push_str(s.as_str());
        }),
    );
    eprintln!("{err:?}");
    println!("RESULT `{buf}`");

    buf
}

//

struct Handler;

#[async_trait]
impl EventHandler for Handler {
    async fn message(&self, ctx: Context, msg: Message) {
        // let Some(bot) = self.bot.get() else { return };
        if msg.author.bot {
            return;
        }

        if !msg.mentions_me(&ctx.http).await.unwrap() && rand::thread_rng().gen_ratio(10, 11) {
            return;
        }

        println!("RUNNING FOR `{}`", msg.content);

        let Ok(permit) = TASKS.try_acquire() else {
            return;
        };

        let messages = msg
            .channel_id
            .messages(&ctx.http, GetMessages::new().limit(15))
            .await
            .unwrap()
            .into_iter()
            .rev()
            .fold((String::new(), None), |(mut acc, mut last_user), msg| {
                let sender = msg.author.name.as_str();
                let sender = if let Some(d) = msg.author.discriminator {
                    format!("{sender}#{d:04}")
                } else {
                    sender.to_string()
                };

                let content = msg.content_safe(ctx.cache().unwrap()).replace('@', "");
                let id = msg.author.id;
                if last_user == Some(id) {
                    writeln!(&mut acc, "{content}").unwrap();
                } else {
                    writeln!(&mut acc, "### {sender}:{content}").unwrap();
                }
                last_user = Some(id);

                (acc, last_user)
            })
            .0;

        let res = async { spawn_blocking(move || prompt(messages)).await.unwrap() };

        let heartbeat = || async {
            loop {
                msg.channel_id.broadcast_typing(&ctx.http).await.unwrap();
                sleep(Duration::from_millis(2000)).await;
            }
        };

        let res = tokio::select! {
            _ = heartbeat() => unreachable!(),
            res = res => res,
        };

        drop(permit);

        if res.is_empty() {
            return;
        }

        if let Err(err) = msg.channel_id.say(&ctx.http, res).await {
            eprintln!("Error sending response {err}");
        }
    }

    async fn ready(&self, _: Context, ready: Ready) {
        println!("logged in as {}", ready.user.name);
        BOT_NAME.get_or_init(|| format!("### {}", ready.user.name));
    }
}

//

#[tokio::main]
async fn main() {
    dotenv().expect(".env file is needed");

    BOT_PERSONA
        .get_or_init(|| env::var("DC_BOT_PERSONA").expect("DC_BOT_PERSONA env var is needed"));

    // Login with a bot token from the environment
    let token = env::var("DC_TOKEN").expect("DC_TOKEN env var is needed");

    // Set gateway intents, which decides what events the bot will be notified about
    let intents = GatewayIntents::GUILD_MESSAGES
        | GatewayIntents::DIRECT_MESSAGES
        | GatewayIntents::MESSAGE_CONTENT;

    // Create a new instance of the Client, logging in as a bot.
    let mut client = Client::builder(&token, intents)
        .event_handler(Handler)
        .await
        .expect("Err creating client");

    // Start listening for events by starting a single shard
    if let Err(why) = client.start().await {
        println!("Client error: {why:?}");
    }
}
