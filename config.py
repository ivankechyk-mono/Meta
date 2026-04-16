from dotenv import load_dotenv
import os

load_dotenv()

# Meta Ads API
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN", "")
META_AD_ACCOUNT   = "act_455699156062655"
META_API_VERSION  = "v21.0"
META_API_BASE     = f"https://graph.facebook.com/{META_API_VERSION}"

# Мітки в назвах кампаній
CAMPAIGN_MARKS = {
    "YO":  "_pr_mpc_reg_YO",
    "FOP": "_pr_mpc_reg_FOP",
    "Acquiring": "_pr_mpc_reg_Acquiring",
}

# ENOT AI (internal monobank Claude proxy)
ENOT_API_KEY = os.getenv("ENOT_API_KEY", "")
ENOT_API_URL = "https://enot.ai.mono.t3zt.com/api/3aw"

# Скорингові параметри
SCORING = {
    "min_impressions":  500,    # мінімум показів для основного рейтингу
    "min_clicks":  50,     # мінімум кліків для CVR
    "weight_ctr":  0.35,
    "weight_cvr":  0.35,
    "weight_cpa":  0.30,
    # Bayesian prior (empirical bayes з середніми по датасету)
    "ctr_prior_n": 1000,
    "cvr_prior_n": 100,
}
