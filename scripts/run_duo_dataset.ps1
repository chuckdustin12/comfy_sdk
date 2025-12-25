param(
  [string]$ComfyUrl = "http://127.0.0.1:8188",
  [string]$ClinLora = "CLIN8-000003.safetensors",
  [string]$ClinToken = "CLIN8",
  [string]$AmberLora = "AMBER8-000005.safetensors",
  [string]$AmberToken = "AMBER6",
  [string]$OutputDir = "DUO/11_DUO",
  [string]$OutputPrefix = "duo_gen_clin8",
  [int]$DuoCount = 10
)

$env:COMFY_URL = $ComfyUrl
$env:CLIN_LORA = $ClinLora
$env:CLIN_TOKEN = $ClinToken
$env:AMBER_LORA = $AmberLora
$env:AMBER_TOKEN = $AmberToken
$env:DUO_OUTPUT_DIR = $OutputDir
$env:DUO_OUTPUT_PREFIX = $OutputPrefix
$env:DUO_COUNT = "$DuoCount"

python scripts\generate_duo_dataset.py
