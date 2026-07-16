"""AST-extract every plugin.add_option(...) from cl-revenue-ops.py."""
import ast, json, sys

tree = ast.parse(open("cl-revenue-ops.py").read())
opts = []
for node in ast.walk(tree):
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_option"):
        continue
    def lit(x):
        try: return ast.literal_eval(x)
        except Exception: return None
    args = [lit(a) for a in node.args]
    kw = {k.arg: lit(k.value) for k in node.keywords}
    name = kw.get("name", args[0] if len(args) > 0 else None)
    default = kw.get("default", args[1] if len(args) > 1 else None)
    desc = kw.get("description", args[2] if len(args) > 2 else "") or ""
    opt_type = kw.get("opt_type", args[3] if len(args) > 3 else "string") or "string"
    dynamic = bool(kw.get("dynamic", False))
    if name:
        opts.append({"name": name, "opt_type": opt_type, "default": default,
                     "description": desc, "dynamic": dynamic})
json.dump(opts, open(sys.argv[1], "w"), indent=1)
print(f"{len(opts)} options -> {sys.argv[1]}")
