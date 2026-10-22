# Jonathan session contracts and compatibility

The physical owner is `studies.session_building.jonathan`. Configuration,
BuildContext, filename parsers and tracking discovery can be imported without
loading the builder's project enrichment. The direct builder now imports the
physical IFdN enrichment owner and its seven preserved source-policy override
files. See [the policy boundary](../../oft_baseline/ifdn_policy/README.md).
Use `from studies.session_building.jonathan.build import build`; the package-level
`build` name can refer to a module or function depending on import order. That
separate lazy-export issue remains unresolved, and a complete empirical cohort
build has not been executed.

Historical `syllable_generation.loaders.jonathan` leaf modules proxy to these
physical owners. Their parent packages retain physical search paths so those
proxies execute. Leaf module/function/class identity is tested in both orders;
parent package identity and direct `studies.behavior` directory-symlink imports
are separate obligations. Three test-created historical class states reconstruct
as the canonical JonathanConfig, BuildContext and ParsedH5 types. This is not
proof that arbitrary historical session/model pickles can be loaded.

Class definitions, parser code and default channel/clock policy are preserved.
Shared stages now read explicit `ctx.cfg`, with historical direct-call fallbacks;
their discovery and numerical algorithms are preserved. Tests independently check timestamp text,
chambers, camera slots, chamber indices, effective channel indices and discovery
rows using synthetic filenames. No actual H5/video contents are opened. Juvenile
configuration now imports the canonical class; its own settings and functions are
unchanged. The IFdN policy files match pinned969c893 exactly; no other cohort's
overrides were substituted. The [unified builder](../README.md) now selects audited
physical owners and transports configuration through stages and process workers.
AMPH/flipped policy recovery, full empirical builders and artifact verification
remain separate work.
