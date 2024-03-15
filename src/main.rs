use std::{convert::Infallible, fmt, num::NonZeroU16, time::Duration};
use std::{env, fmt::Write};

use dashmap::DashMap;
use dotenv::dotenv;
use llm::{
    conversation_inference_callback, load_progress_callback_stdout, models::Llama,
    InferenceParameters, InferenceRequest, KnownModel, ModelParameters, Prompt,
};
use once_cell::sync::{Lazy, OnceCell};
use rand::Rng;
use serenity::{
    all::{
        ChannelId, Command, CommandOptionType, CreateCommand, CreateCommandOption,
        CreateInteractionResponse, CreateInteractionResponseMessage, GetMessages, Integration,
        Interaction, Message, Ready, ResolvedValue, User,
    },
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
            use_gpu: true,
            ..<_>::default()
        },
        load_progress_callback_stdout,
    )
    .unwrap()
});
static BOT_NAME: OnceCell<String> = OnceCell::new();
static BOT_PERSONA: OnceCell<String> = OnceCell::new();

//

fn run_prompt(prompt: String) -> String {
    let mut rng = rand::thread_rng();
    let mut sess = BOT.start_session(<_>::default());
    let mut buf = String::new();

    // let prompt = format!(
    //     "{persona}\n\nContext:\n\
    //         {messages}\
    //         {bot_name}:"
    // );

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

struct UserFmt<'a>(&'a User);

impl fmt::Display for UserFmt<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let name = self.0.name.as_str();
        if let Some(d) = self.0.discriminator {
            write!(f, "{name}#{d:04}")
        } else {
            write!(f, "{name}")
        }
    }
}

//

struct Handler {
    personas: DashMap<ChannelId, String>,
    bot: OnceCell<User>,
}

impl Handler {
    pub fn new() -> Self {
        Self {
            personas: DashMap::new(),
            bot: OnceCell::new(),
        }
    }
}

#[async_trait]
impl EventHandler for Handler {
    async fn message(&self, ctx: Context, msg: Message) {
        if msg.author.bot {
            return;
        }

        let Some(me) = self.bot.get() else {
            return;
        };

        if !msg.mentions_me(&ctx.http).await.unwrap() && rand::thread_rng().gen_ratio(10, 11) {
            return;
        }

        let Ok(permit) = TASKS.try_acquire() else {
            return;
        };

        let persona = self.personas.get(&msg.channel_id);
        let persona = persona
            .as_ref()
            .map_or(BOT_PERSONA.get().unwrap().as_str(), |s| s.value().as_str());

        println!("RUNNING FOR `{}`", msg.content);

        let mut prompt = String::new();

        writeln!(&mut prompt, "{persona}\n\nContext:").unwrap();

        msg.channel_id
            .messages(&ctx.http, GetMessages::new().limit(5))
            .await
            .unwrap()
            .into_iter()
            .rev()
            .fold(None, |last_user, msg| {
                let content = msg.content_safe(ctx.cache().unwrap()).replace('@', "");
                let id = msg.author.id;
                if last_user == Some(id) {
                    // combine subsequent messages from the same user
                    writeln!(&mut prompt, "{content}").unwrap();
                } else {
                    writeln!(&mut prompt, "### {}: {content}", UserFmt(&msg.author)).unwrap();
                }
                Some(id)
            });

        write!(&mut prompt, "### {}: ", UserFmt(me)).unwrap();

        let res = async { spawn_blocking(move || run_prompt(prompt)).await.unwrap() };

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

    async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
        let Interaction::Command(cmd) = interaction else {
            return;
        };

        if let "persona" = cmd.data.name.as_str() {
            let cmds = cmd.data.options();
            let response = if let Some(ResolvedValue::String(new)) = cmds.first().map(|v| &v.value)
            {
                self.personas.insert(cmd.channel_id, new.to_string());
                "done".to_string()
            } else {
                let persona = self.personas.get(&cmd.channel_id);
                let persona = persona
                    .as_ref()
                    .map_or(BOT_PERSONA.get().unwrap().as_str(), |s| s.as_str());
                persona.to_string()
            };

            if let Err(err) = cmd
                .create_response(
                    &ctx.http,
                    CreateInteractionResponse::Message(
                        CreateInteractionResponseMessage::new().content(response),
                    ),
                )
                .await
            {
                eprintln!("cannot respond: {err:?}");
            }
        };
    }

    async fn ready(&self, ctx: Context, ready: Ready) {
        for guild in ready.guilds.into_iter().filter(|g| g.unavailable) {
            // println!("guild_id={} unavailable={}", guild.id, guild.unavailable);
            println!(
                "registering commands for {:?}",
                guild.id.name(ctx.cache().unwrap())
            );
            _ = guild.id.set_commands(&ctx.http, vec![
                CreateCommand::new("persona")
                    .description("Change the bot's persona. Refer to the bot by its name. Example: \"A chat between discord users.\"")
                    .add_option(CreateCommandOption::new(CommandOptionType::String, "persona", "The new persona"))
            ]).await;
        }

        Command::set_global_commands(&ctx.http, vec![])
            .await
            .unwrap();

        println!("logged in as {}", ready.user.name);
        self.bot.get_or_init(|| (*ready.user).clone());
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
        .event_handler(Handler::new())
        .await
        .expect("Err creating client");

    // Start listening for events by starting a single shard
    if let Err(why) = client.start().await {
        println!("Client error: {why:?}");
    }
}
