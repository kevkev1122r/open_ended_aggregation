"""
Emit the `az` commands that provision the pool. Prints only -- never executes.

WHY A SCRIPT AND NOT A LIST OF CLICKS
  Eight deployments, each needing a model name, a version, a publisher format, a
  SKU and a capacity. Three of the eight have a catalogue ID that is the model
  name and version CONCATENATED (DeepSeek-V4-Flash-2026-04-23 is name
  "DeepSeek-V4-Flash" + version "2026-04-23"), so pasting the catalogue ID into
  --model-name fails with a misleading "model not found". The splits below were
  read out of the live 400 responses, not guessed.

THE NAMING RULE THAT MATTERS
  After deployment you call the DEPLOYMENT NAME, not the model name. Name each
  deployment exactly as its entry in run_v2.AZURE_MODELS and the code needs no
  mapping layer. Deviate and you get 404s that look like the model is missing.

USAGE
  ./venv/bin/python deploy_models.py --resource <name> --group <rg>

  Publisher formats vary per model and Azure is the only authority on them, so
  resolve them first and feed them back in:

    az cognitiveservices model list -l <region> -o json > /tmp/models.json
    ./venv/bin/python deploy_models.py --resource <r> --group <g> --models /tmp/models.json
"""
import argparse, json, sys

# catalogue id -> (model name, version). Read from live 400 responses 2026-08-13.
POOL = {
    # gpt-5.4 answered without a deployment on 2026-08-13 and STOPPED once the
    # other deployments were created: "Insufficient quota available for instant
    # inference" / deployment_disabled. Deploymentless is a shared instant pool,
    # not reserved capacity -- it is not a foundation to build a run on.
    "gpt-5.4":                       ("gpt-5.4", "2026-03-05"),
    "grok-4.3":                      ("grok-4.3", "1"),
    "DeepSeek-V4-Flash-2026-04-23":  ("DeepSeek-V4-Flash", "2026-04-23"),
    "Kimi-K2.6-2026-04-20":          ("Kimi-K2.6", "2026-04-20"),
    "Cohere-command-a-plus-05-2026": ("Cohere-command-a-plus-05-2026", "1"),
    "MAI-Thinking-1-2026-06-01":     ("MAI-Thinking-1", "2026-06-01"),
}
# gpt-5.4 is absent on purpose: it already answers without a deployment.
# claude-sonnet-5, mistral-medium-3-5 and Llama-4-Maverick are absent because the
# subscription has 0/0 TPM for them -- see the note in run_v2.AZURE_MODELS for why
# their older siblings were not substituted in.


def formats_from(path):
    """Resolve --model-format exactly, from `az cognitiveservices model list`."""
    out = {}
    for e in json.load(open(path)):
        m = e.get("model", e)
        n, v, f = m.get("name"), str(m.get("version", "")), m.get("format")
        if n and f:
            out[(n, v)] = f
            out.setdefault((n, None), f)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resource", required=True, help="AI Services resource name")
    p.add_argument("--group", required=True, help="resource group")
    p.add_argument("--sku", default="GlobalStandard")
    p.add_argument("--capacity", type=int, default=200,
                   help="thousands of tokens/min. 9 models x 27k calls at MAXW=10 "
                        "needs headroom; too low throttles into 429 storms.")
    p.add_argument("--models", help="json from `az cognitiveservices model list`")
    a = p.parse_args()

    fmts = formats_from(a.models) if a.models else {}
    if not fmts:
        print("# NOTE: --model-format left as <FORMAT> below. Resolve it with:")
        print(f"#   az cognitiveservices account show -n {a.resource} -g {a.group} "
              "--query location -o tsv")
        print("#   az cognitiveservices model list -l <region> -o json > /tmp/models.json")
        print("# then re-run this with --models /tmp/models.json\n")

    for dep, (name, ver) in POOL.items():
        fmt = fmts.get((name, ver)) or fmts.get((name, None)) or "<FORMAT>"
        # --name is the DEPLOYMENT name and must equal the catalogue id, because
        # that is the string run_v2.AZURE_MODELS sends as "model".
        print(f"az cognitiveservices account deployment create \\\n"
              f"  --name {a.resource} --resource-group {a.group} \\\n"
              f"  --deployment-name '{dep}' \\\n"
              f"  --model-name '{name}' --model-version '{ver}' \\\n"
              f"  --model-format '{fmt}' \\\n"
              f"  --sku-name {a.sku} --sku-capacity {a.capacity}\n")

    print("# verify, then price them, then preflight:")
    print("#   ./venv/bin/python azure_backend.py            # deployments now non-empty?")
    print("#   # fill AZ.PRICE from the pricing page for your region")
    print("#   ./venv/bin/python run_v2.py preflight")


if __name__ == "__main__":
    main()
