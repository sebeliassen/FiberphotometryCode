# Jonathan session contracts and compatibility

The physical owner is `studies.session_building.jonathan`. Configuration,
BuildContext, filename parsers and tracking discovery can be imported without
loading the builder's project enrichment. The direct builder now imports the
physical IFdN enrichment owner and its seven preserved source-policy override
files. See [the policy boundary](../../oft_baseline/ifdn_policy/README.md).
Use the stable package callable:

```python
from studies.session_building.jonathan import BuildContext, build_sessions
```

`build_sessions` is the exact existing `studies.session_building.jonathan.build.build`
function, with the same signature, context, explicit-step behavior and pickle
qualified name. It is loaded on request and stays callable across canonical and
historical package/leaf import orders. Importing configuration or inspecting
`dir(package)` does not load the builder or its project policy.

The explicit leaf API `from studies.session_building.jonathan.build import build`
continues to work. The old package-level `build` export remains available for
external notebook compatibility: it can still refer to a function or module
depending on import order. Do not use that spelling in new code or assume this
additive API removes its ambiguity. No maintained Python/notebook package-export
caller was found in the source audit; that does not establish external usage or
authorize removing the historical name. A complete empirical cohort build has
not been executed.

Historical `syllable_generation.loaders.jonathan` leaf modules proxy to these
physical owners. Their parent packages retain physical search paths so those
proxies execute. Leaf module/function/class identity is tested in both orders;
parent package identity and direct `studies.behavior` directory-symlink imports
are separate obligations. Three test-created historical class states reconstruct
as the canonical JonathanConfig, BuildContext and ParsedH5 types. This is not
proof that arbitrary historical session/model pickles can be loaded.

The historical package also exposes the same stable `build_sessions` callable
through its existing delegation. Package objects and filesystem search paths stay
distinct so physical leaf proxies still execute. No global import finder, custom
module type, wrapper function or eager builder import is introduced.

Class definitions, parser code and default channel/clock policy are preserved.
Shared stages now read explicit `ctx.cfg`, with historical direct-call fallbacks;
their discovery and numerical algorithms are preserved. Tests independently check timestamp text,
chambers, camera slots, chamber indices, effective channel indices and discovery
rows using synthetic filenames. No actual H5/video contents are opened. Juvenile
configuration now imports the canonical class; its own settings and functions are
unchanged. The IFdN policy files match pinned969c893 exactly; no other cohort's
overrides were substituted. The [unified builder](../README.md) now selects audited
physical owners and transports configuration through stages and process workers.
AMPH source/metadata has a separate recovered owner; its scientific execution is
still gated. Flipped provenance, full empirical builders and artifact verification
remain separate work.
