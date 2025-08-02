use clap::Parser;
use hyper::server::conn::http1;
use hyper_util::rt::TokioIo;
use log;
use router::Logger;
use router::RequestHandler;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;

static LOGGER: Logger = Logger;

#[derive(Parser)]
#[command(name = "vllm-router")]
struct Args {
    /// router's listening port
    #[arg(short, long, default_value_t = 7999)]
    port: u16,

    /// Path to JSON configuration file
    #[arg(short, long, value_name = "FILE")]
    config: std::path::PathBuf,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _ = log::set_logger(&LOGGER).map(|()| log::set_max_level(log::LevelFilter::Info));
    let args = Args::parse();

    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));
    let listener = TcpListener::bind(addr).await?;

    let handler = match RequestHandler::new(&args.config) {
        Ok(h) => h,
        Err(e) => {
            log::error!("{}", e);
            return Err(e.into());
        }
    };
    let handler = Arc::new(handler);

    log::info!("vLLM router is ready to serve on port {}", args.port);
    loop {
        let (stream, _) = listener.accept().await?;
        let io = TokioIo::new(stream);

        let h = handler.clone();
        tokio::task::spawn(async move {
            if let Err(err) = http1::Builder::new().serve_connection(io, h).await {
                log::error!("Error serving connection: {}", err);
            }
        });
    }
}
