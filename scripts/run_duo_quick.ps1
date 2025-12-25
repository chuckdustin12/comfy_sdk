param(
  [string]$ComfyUrl = "http://127.0.0.1:8000",
  [string]$ClinLora = "CLIN8-000003.safetensors",
  [string]$ClinToken = "CLIN8",
  [string]$AmberLora = "AMBER8-000005.safetensors",
  [string]$AmberToken = "AMBER6",
  [string]$OutputDir = "generated/duo_quick",
  [string]$OutputPrefix = "duo_quick_clin8",
  [string]$Positive,
  [string]$Negative,
  [int]$Seed,
  [int]$Steps = 30,
  [double]$Cfg = 3.5,
  [string]$SamplerName = "dpmpp_2m_sde",
  [string]$Scheduler = "karras",
  [double]$Denoise = 1.0,
  [int]$Width = 1152,
  [int]$Height = 1024,
  [int]$BatchSize = 1
)

$env:COMFY_URL = $ComfyUrl
$env:CLIN_LORA = $ClinLora
$env:CLIN_TOKEN = $ClinToken
$env:AMBER_LORA = $AmberLora
$env:AMBER_TOKEN = $AmberToken
$env:DUO_OUTPUT_DIR = $OutputDir
$env:DUO_OUTPUT_PREFIX = $OutputPrefix
$env:STEPS = "$Steps"
$env:CFG = "$Cfg"
$env:SAMPLER_NAME = $SamplerName
$env:SCHEDULER = $Scheduler
$env:DENOISE = "$Denoise"
$env:WIDTH = "$Width"
$env:HEIGHT = "$Height"
$env:BATCH_SIZE = "$BatchSize"

if ($PSBoundParameters.ContainsKey("Seed")) { $env:SEED = "$Seed" }
if ($PSBoundParameters.ContainsKey("Positive")) { $env:POSITIVE_PROMPT = $Positive }
if ($PSBoundParameters.ContainsKey("Negative")) { $env:NEGATIVE_PROMPT = $Negative }

python scripts\generate_duo_quick.py
