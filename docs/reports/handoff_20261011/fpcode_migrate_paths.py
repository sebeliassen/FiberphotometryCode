"""Bounded, reversible migration of study imports and filesystem references."""
import argparse
import ast
import json
import os
from pathlib import Path, PurePath

ROOT = Path('/home/fsp585/Desktop/FiberphotometryCode')
PREFIX = str(ROOT)
MOVES = []
for row in (ROOT / 'registry/STRUCTURE_MOVES.tsv').read_text().splitlines()[1:]:
    old, new, *_ = row.split('\t')
    MOVES.append((old, new))
MODELS = [p.name for p in (ROOT / 'data/models').iterdir() if p.is_dir()]
LEGACY = [('syllable_generation/multiview_top_bot_experiments/' + name, 'data/models/' + name) for name in MODELS]
LEGACY += [
    ('syllable_generation/data', 'data/sessions'),
    ('karen_project', 'data/raw/karen_pff_vs_monomer'),
    ('src_oft', 'data/raw/oft_baseline'),
    ('src_mono_inter', 'data/raw/juvenile_intruder'),
    ('monoamine_interactions', 'studies/monoamine_interactions'),
    ('syllable_generation/analysis/karen_syllable_analysis/output', 'outputs/karen_pff_vs_monomer/legacy'),
    ('syllable_generation/analysis/karen_syllable_analysis', 'studies/karen_pff_vs_monomer/analysis'),
    ('syllable_generation/analysis/ulrik_kinematics/output', 'data/analysis_cache/ulrik_kinematics_output'),
    ('syllable_generation/analysis/ulrik_kinematics/cache', 'data/analysis_cache/ulrik_kinematics_cache'),
    ('syllable_generation/analysis/ulrik_kinematics', 'studies/kinematics_ulrik'),
    ('syllable_generation/analysis/language_models/output', 'data/analysis_cache/language_models_output'),
    ('syllable_generation/analysis/language_models', 'studies/language_models'),
    ('syllable_generation/analysis/sequence_viz', 'studies/sequence_viz'),
    ('syllable_generation/analysis/replications', 'studies/replications_datta'),
    ('syllable_generation/analysis/deep_research_experiments', 'studies/deep_research_experiments'),
    ('syllable_generation/videogen', 'studies/multimodal_movies'),
    ('syllable_generation/loaders', 'studies/session_building'),
    ('syllable_generation/projects/jonathan_openfield_example', 'studies/oft_baseline/ifdn_policy'),
    ('syllable_generation/projects/jonathan_openfield_amph', 'studies/oft_baseline/ifdn_policy'),
    ('syllable_generation/syllabizers/keypoint_moseq_syll', 'vendor/keypoint_moseq_syll'),
    ('syllable_generation/multiview_top_bot_experiments/data', 'data/fused/multiview'),
    ('syllable_generation/multiview_top_bot_experiments/fused_h5_full_cohort', 'data/fused/multiview/fused_h5_full_cohort'),
    ('syllable_generation/multiview_top_bot_experiments/fused_h5_continuous_calibrated', 'data/fused/multiview/fused_h5_continuous_calibrated'),
    ('syllable_generation/multiview_top_bot_experiments/session_caches', 'data/sessions/multiview'),
    ('syllable_generation/multiview_top_bot_experiments/04_reports_and_figures/visualizations', 'data/analysis_cache/multiview_visualizations/visualizations'),
    ('syllable_generation/multiview_top_bot_experiments', 'studies/multiview_experiments'),
    ('syllable_generation/analysis', 'studies/session_model_exports'),
    ('syllable_generation', 'compat/syllable_generation'),
]
RULES = sorted(MOVES + LEGACY, key=lambda p: len(p[0]), reverse=True)
MODULES = sorted([(a.replace('/', '.'), b.replace('/', '.')) for a,b in MOVES if a.startswith('studies/')], key=lambda p: len(p[0]), reverse=True)

def redirect(value):
    absolute = value.startswith(PREFIX)
    rel = value[len(PREFIX):].lstrip('/') if absolute else value
    for _ in range(4):
        changed = False
        for old,new in RULES:
            boundary = rel == old or rel.startswith(old + '/')
            visual = old.endswith('/visualizations') and rel.startswith(old)
            if boundary or visual:
                replacement = new + rel[len(old):]
                if replacement != rel:
                    rel = replacement; changed = True
                break
        if not changed:
            break
    return str(ROOT / rel) if absolute else rel

def source_imports(text):
    for old,new in MODULES:
        text = text.replace(old + '.', new + '.').replace(old + ' import', new + ' import')
        for quote in ('"', "'"):
            text = text.replace(quote + old + quote, quote + new + quote)
    # Consumers and physical historical adapters reference the live tree directly.
    return text.replace('archive/compat', 'compat')

def original_path(path):
    rel = str(path.relative_to(ROOT))
    for old,new in sorted(MOVES, key=lambda p: len(p[1]), reverse=True):
        if old.startswith('studies/') and (rel == new or rel.startswith(new + '/')):
            return str(ROOT / (old + rel[len(new):]))
    return str(path)

def evaluate(node, env):
    if isinstance(node, ast.Constant) and isinstance(node.value, (str,int)):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.BinOp):
        left,right = evaluate(node.left, env),evaluate(node.right, env)
        if isinstance(left,str) and isinstance(right,str):
            if isinstance(node.op,ast.Div): return str(PurePath(left)/right)
            if isinstance(node.op,ast.Add): return left+right
    if isinstance(node,ast.Call):
        name = ast.unparse(node.func)
        if name in {'get_repo_root','paths.get_repo_root'}: return PREFIX
        if name in {'Path','str','os.path.abspath','os.path.realpath','os.path.expanduser'} and node.args:
            return evaluate(node.args[0],env)
        if name in {'resolve_path','resolve_data_path'} and node.args:
            arg=evaluate(node.args[0],env)
            return str(PurePath(PREFIX)/arg) if isinstance(arg,str) else None
        if name == 'os.path.join':
            vals=[evaluate(n,env) for n in node.args]
            return str(PurePath(*vals)) if vals and all(isinstance(v,str) for v in vals) else None
        if isinstance(node.func,ast.Attribute) and node.func.attr in {'resolve','absolute'}:
            return evaluate(node.func.value,env)
    if isinstance(node,ast.Attribute) and node.attr == 'parent':
        value=evaluate(node.value,env)
        return str(PurePath(value).parent) if isinstance(value,str) else None
    if isinstance(node,ast.Subscript) and isinstance(node.value,ast.Attribute) and node.value.attr=='parents':
        value=evaluate(node.value.value,env); index=evaluate(node.slice,env)
        if isinstance(value,str) and isinstance(index,int):
            try: return str(PurePath(value).parents[index])
            except IndexError: return None
    if isinstance(node,ast.JoinedStr):
        vals=[evaluate(n.value,env) if isinstance(n,ast.FormattedValue) else evaluate(n,env) for n in node.values]
        return ''.join(map(str,vals)) if all(v is not None for v in vals) else None
    return None

def migrate(text,path):
    text=source_imports(text)
    tree=ast.parse(text)
    env={'REPO_ROOT':PREFIX,'ROOT_DIR':PREFIX,'repo_root':PREFIX,'__file__':original_path(path)}
    for stmt in tree.body:
        if isinstance(stmt,(ast.Assign,ast.AnnAssign)):
            val=evaluate(stmt.value,env)
            targets=stmt.targets if isinstance(stmt,ast.Assign) else [stmt.target]
            for target in targets:
                if isinstance(target,ast.Name) and val is not None: env[target.id]=val
    lines=text.splitlines(keepends=True)
    starts=[]; offset=0
    for line in lines: starts.append(offset); offset+=len(line)
    edits=[]; helpers=set()
    def bounds(node):
        start_col=len(lines[node.lineno-1].encode()[:node.col_offset].decode())
        end_col=len(lines[node.end_lineno-1].encode()[:node.end_col_offset].decode())
        return starts[node.lineno-1]+start_col,starts[node.end_lineno-1]+end_col
    def expression(value, string=False):
        rel=value[len(PREFIX):].lstrip('/') if value.startswith(PREFIX) else value
        helper='resolve_data_path' if rel.startswith('data/') else 'get_repo_root'
        helpers.add(helper)
        result=f'resolve_data_path({rel!r})' if helper=='resolve_data_path' else ('get_repo_root()' if not rel else f'(get_repo_root() / {rel!r})')
        return f'str({result})' if string else result
    # A legacy directory mixed scripts and datasets. Rewrite crossings explicitly,
    # e.g. EXP_DIR/cohort becomes data/models/cohort, retaining subsequent joins.
    for node in ast.walk(tree):
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div):
            value=evaluate(node,env); left=evaluate(node.left,env); right=evaluate(node.right,env)
            if isinstance(value,str) and value.startswith(PREFIX) and redirect(value)!=value:
                mapped=redirect(value)
                mapped_right=redirect(right) if isinstance(right,str) and '/' in right else right
                parent_join=str(PurePath(redirect(left))/mapped_right) if isinstance(left,str) and isinstance(mapped_right,str) else None
                if parent_join != mapped:
                    edits.append((*bounds(node),expression(mapped)))
        elif isinstance(node,ast.Call) and ast.unparse(node.func) in {'resolve_path','resolve_data_path','os.path.join'}:
            value=evaluate(node,env)
            if isinstance(value,str) and value.startswith(PREFIX) and redirect(value)!=value:
                edits.append((*bounds(node),expression(redirect(value),string=ast.unparse(node.func)=='os.path.join')))
        elif isinstance(node,ast.Constant) and isinstance(node.value,str):
            value=node.value
            if value == PREFIX or value == PREFIX+'/':
                replacement=expression(PREFIX,string=True)
                if value.endswith('/'):replacement+=" + '/'"
                edits.append((*bounds(node),replacement))
            elif value.startswith(PREFIX+'/') and '\n' not in value:
                replacement=expression(redirect(value),string=True)
                if value.endswith('/'):replacement+=" + '/'"
                edits.append((*bounds(node),replacement))
            else:
                mapped=redirect(value) if '/' in value else value
                if mapped != value:
                    edits.append((*bounds(node),repr(mapped)))
    # Root calculation depths can change when a study gains one canonical home.
    for stmt in tree.body:
        if isinstance(stmt,(ast.Assign,ast.AnnAssign)):
            targets=stmt.targets if isinstance(stmt,ast.Assign) else [stmt.target]
            if any(isinstance(n,ast.Name) and n.id.lower() in {'repo_root','root_dir','project_root'} for n in targets):
                if evaluate(stmt.value,env)==PREFIX and '__file__' in ast.unparse(stmt.value):
                    edits.append((*bounds(stmt.value),expression(PREFIX)))
    selected=[]
    for edit in sorted(edits,key=lambda x:(x[0],-x[1])):
        if not selected or edit[0]>=selected[-1][1]: selected.append(edit)
    for start,end,replacement in reversed(selected): text=text[:start]+replacement+text[end:]
    if helpers:
        fresh=ast.parse(text);at=0
        existing={alias.asname or alias.name for stmt in fresh.body if isinstance(stmt,ast.ImportFrom) and stmt.module=='fpcode.common.paths' for alias in stmt.names}
        helpers-=existing
        for stmt in fresh.body:
            if isinstance(stmt,ast.Expr) and isinstance(stmt.value,ast.Constant) and isinstance(stmt.value.value,str) or isinstance(stmt,ast.ImportFrom) and stmt.module=='__future__':
                at=stmt.end_lineno
            else: break
        current=text.splitlines(keepends=True)
        if helpers:current.insert(at,'from fpcode.common.paths import '+', '.join(sorted(helpers))+'\n')
        text=''.join(current)
    ast.parse(text)
    return text

def main():
    parser=argparse.ArgumentParser();parser.add_argument('project');parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();base=ROOT/args.project
    assert base.is_dir() and not base.is_symlink(),base
    count=0
    for folder,dirs,files in os.walk(base,followlinks=False):
        dirs[:]=[d for d in dirs if d!='__pycache__' and not (Path(folder)/d).is_symlink()]
        for name in files:
            path=Path(folder)/name
            if path.suffix!='.py' or path.is_symlink():continue
            before=path.read_text()
            try: after=migrate(before,path)
            except (SyntaxError,ValueError) as error:
                print('SKIP',path.relative_to(ROOT),error);continue
            if before!=after:
                count+=1
                if args.apply:
                    backup=Path('/tmp/fpcode-path-migration-originals')/path.relative_to(ROOT)
                    backup.parent.mkdir(parents=True,exist_ok=True)
                    if not backup.exists():backup.write_text(before)
                    path.write_text(after)
    print(args.project, count, 'changed' if args.apply else 'planned')

if __name__=='__main__':main()
