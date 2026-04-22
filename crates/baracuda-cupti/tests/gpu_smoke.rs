//! CUPTI presence + version probe.

#[test]
fn cupti_version_query() {
    match baracuda_cupti::version() {
        Ok(v) => eprintln!("CUPTI version: {v}"),
        Err(e) => eprintln!("CUPTI not available on this host: {e:?}"),
    }
}
