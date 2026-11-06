// Build glue for the optional coro_rpc Conductor transport (`conductor-rpc`).
//
// Active only under that cargo feature. It statically links the Mooncake C++
// Conductor client (`libconductor_cpp_core.a`), which speaks coro_rpc to the
// Conductor, and adds the runtime search paths for the shared libraries the
// archive references (`libasio.so` from the Mooncake build tree and the
// system-provided zmq/glog/jsoncpp).
//
// The library location must be provided explicitly (`MOONCAKE_CONDUCTOR_LIB_DIR`,
// or `MOONCAKE_BUILD_DIR` pointing at a Mooncake build tree). When it cannot be
// found we emit a warning and skip linking instead of failing the build: CI runs
// `cargo clippy --all-features` on machines without a Mooncake tree, and only a
// real `cargo build`/`test` with the feature needs the archive.
//
// Without the feature no C++ is referenced and the build stays dependency-free.

use std::env;
use std::path::PathBuf;

fn main() {
    if env::var_os("CARGO_FEATURE_CONDUCTOR_RPC").is_none() {
        return;
    }

    let Some(lib_dir) = env::var_os("MOONCAKE_CONDUCTOR_LIB_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            env::var_os("MOONCAKE_BUILD_DIR")
                .map(|dir| PathBuf::from(dir).join("mooncake-conductor"))
        })
    else {
        println!(
            "cargo:warning=conductor-rpc: set MOONCAKE_CONDUCTOR_LIB_DIR (or MOONCAKE_BUILD_DIR) \
             to the Mooncake build tree; the feature will not link"
        );
        return;
    };

    let archive = lib_dir.join("libconductor_cpp_core.a");
    if !archive.is_file() {
        println!(
            "cargo:warning=conductor-rpc: {} not found; the feature will not link",
            archive.display()
        );
        return;
    }

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

    // `libasio.so` (ASIO_SEPARATE_COMPILATION build of yalantinglibs) lives next
    // to the conductor archive in the Mooncake build tree.
    if let Some(tree_root) = lib_dir.parent() {
        let common = tree_root.join("mooncake-common");
        if common.is_dir() {
            println!("cargo:rustc-link-search=native={}", common.display());
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", common.display());
        }
    }

    // Mirrors the link set of mooncake-conductor/rust/build.rs. `asio` is
    // required (ASIO_SEPARATE_COMPILATION, symbols in libasio.so); `ibverbs` and
    // `gflags` are pulled in by yalantinglibs; the rest are used by the client
    // core. Unused shared libraries are dropped by `--as-needed`.
    for lib in [
        "conductor_cpp_core",
        "asio",
        "ibverbs",
        "gflags",
        "zmq",
        "crypto",
        "glog",
        "jsoncpp",
        "stdc++",
        "pthread",
        "m",
    ] {
        println!("cargo:rustc-link-lib={lib}");
    }

    println!("cargo:rerun-if-env-changed=MOONCAKE_CONDUCTOR_LIB_DIR");
    println!("cargo:rerun-if-env-changed=MOONCAKE_BUILD_DIR");
    println!("cargo:rerun-if-changed={}", archive.display());
}
