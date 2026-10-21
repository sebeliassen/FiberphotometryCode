# Jonathan session contracts and compatibility

The physical owner is `studies.session_building.jonathan`. Configuration,
BuildContext, filename parsers and tracking discovery can be imported without
loading the builder's project enrichment. The public package still advertises
`build`; accessing it explicitly imports its implementation. Its old enrichment
namespace remains unresolved, so this change does not establish a working full
cohort builder.

Historical `syllable_generation.loaders.jonathan` leaf modules proxy to these
physical owners. Their parent packages retain physical search paths so those
proxies execute. Leaf module/function/class identity is tested in both orders;
parent package identity and direct `studies.behavior` directory-symlink imports
are separate obligations. Three test-created historical class states reconstruct
as the canonical JonathanConfig, BuildContext and ParsedH5 types. This is not
proof that arbitrary historical session/model pickles can be loaded.

Class definitions, parser code, default channel/clock policy and tracking discovery
are byte-identical to the previous main. Tests independently check timestamp text,
chambers, camera slots, chamber indices, effective channel indices and discovery
rows using synthetic filenames. No actual H5/video contents are opened. Juvenile
configuration now imports the canonical class; its own settings and functions are
unchanged. Karen/monoamine policy recovery, enrichment/override input resolution,
full builders and empirical artifact verification remain separate work.
