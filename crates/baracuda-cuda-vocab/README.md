# baracuda-cuda-vocab

Owner source for the `cuda:` KISS section 6.8 namespace vocabulary.

The closed sm-token set, its base and `a` classes, and the two admission relations. Driver-free: it consumes `unpopped-vocab`'s `ArchSku` codec as the byte source of truth for token spellings.
Note that the open `cuda:smNN` target model accepts tokens outside this closed set (for example `cuda:sm61`); see the repository's `docs/sm61-parse-emit-gap-analysis.md`.

Part of [baracuda](https://github.com/ciresnave/baracuda); licensed under MIT OR Apache-2.0 (see `LICENSE-MIT`, `LICENSE-APACHE`).
