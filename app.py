# ============================================================
# BIRD DETECTOR ULTRA - PREMIUM UI
# 182 BIRD SPECIES AI
# REAL-TIME STYLE FRONTEND
# ============================================================

import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import time
import requests
from io import BytesIO

from tensorflow.keras.models import load_model
from audio_recorder_streamlit import audio_recorder

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Bird Detector Ultra",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(
        135deg,
        #0B0F19,
        #121826,
        #1C2333
    );
    color: white;
}

/* =========================================================
TITLE
========================================================= */

.main-title {
    text-align: center;
    font-size: 65px;
    font-weight: 700;
    background: linear-gradient(
        90deg,
        #00F5A0,
        #00D9F5,
        #7F5AF0
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: -30px;
}

.subtitle {
    text-align: center;
    color: #B0B7C3;
    font-size: 20px;
    margin-bottom: 40px;
}

/* =========================================================
GLASS CARDS
========================================================= */

.glass-card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);

    border: 1px solid rgba(255,255,255,0.1);

    border-radius: 20px;

    padding: 25px;

    margin-bottom: 20px;

    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* =========================================================
PREDICTION CARDS
========================================================= */

.prediction-card {

    background: linear-gradient(
        135deg,
        rgba(0,245,160,0.15),
        rgba(0,217,245,0.15)
    );

    border-radius: 18px;

    padding: 20px;

    margin-bottom: 18px;

    border: 1px solid rgba(255,255,255,0.08);

    transition: 0.3s;
}

.prediction-card:hover {

    transform: scale(1.02);

    box-shadow: 0 0 20px rgba(0,245,160,0.2);
}

/* =========================================================
METRIC CARDS
========================================================= */

.metric-card {

    background: rgba(255,255,255,0.05);

    border-radius: 16px;

    padding: 18px;

    text-align: center;

    border: 1px solid rgba(255,255,255,0.08);
}

/* =========================================================
SIDEBAR
========================================================= */

[data-testid="stSidebar"] {

    background: #0F172A;

    border-right: 1px solid rgba(255,255,255,0.08);
}

/* =========================================================
BUTTON
========================================================= */

.stButton>button {

    width: 100%;

    background: linear-gradient(
        90deg,
        #00F5A0,
        #00D9F5
    );

    color: black;

    border: none;

    padding: 14px;

    border-radius: 14px;

    font-weight: 700;

    font-size: 16px;

    transition: 0.3s;
}

.stButton>button:hover {

    transform: scale(1.02);

    box-shadow: 0 0 25px rgba(0,245,160,0.3);
}

/* =========================================================
UPLOAD BOX
========================================================= */

[data-testid="stFileUploader"] {

    background: rgba(255,255,255,0.04);

    border-radius: 18px;

    padding: 15px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_ai():

    model = load_model(
        "bird_detector_ultra_v4.keras"
    )

    with open(
        "bird_detector_ultra_v4_encoder.pkl",
        "rb"
    ) as f:

        encoder = pickle.load(f)

    return model, encoder

model, encoder = load_ai()

# ============================================================
# BIRD CODE → REAL BIRD NAME
# ============================================================

bird_names = {

    "asbfly": {
        "display": "Asian Brown Flycatcher",
        "wiki": "Asian_brown_flycatcher"
    },

    "ashdro1": {
        "display": "Ashy Drongo",
        "wiki": "Ashy_drongo"
    },

    "ashpri1": {
        "display": "Ashy Prinia",
        "wiki": "Ashy_prinia"
    },

    "ashwoo2": {
        "display": "Ashy Woodswallow",
        "wiki": "Ashy_woodswallow"
    },

    "asikoe2": {
        "display": "Asian Koel",
        "wiki": "Asian_koel"
    },

    "asiope1": {
        "display": "Asian Openbill",
        "wiki": "Asian_openbill"
    },

    "aspfly1": {
        "display": "Ashy-crowned Sparrow-Lark",
        "wiki": "Ashy-crowned_sparrow-lark"
    },

    "aspswi1": {
        "display": "Asian Palm Swift",
        "wiki": "Asian_palm_swift"
    },

    "barfly1": {
        "display": "Barred Flycatcher-shrike",
        "wiki": "Barred_flycatcher-shrike"
    },

    "barswa": {
        "display": "Barn Swallow",
        "wiki": "Barn_swallow"
    },

    "bcnher": {
        "display": "Black-crowned Night Heron",
        "wiki": "Black-crowned_night_heron"
    },

    "bkcbul1": {
        "display": "Black-crested Bulbul",
        "wiki": "Black-crested_bulbul"
    },

    "bkrfla1": {
        "display": "Black-rumped Flameback",
        "wiki": "Black-rumped_flameback"
    },

    "bkskit1": {
        "display": "Black-backed Shrike",
        "wiki": "Long-tailed_shrike"
    },

    "bkwsti": {
        "display": "Black-winged Stilt",
        "wiki": "Black-winged_stilt"
    },

    "bladro1": {
        "display": "Black Drongo",
        "wiki": "Black_drongo"
    },

    "blaeag1": {
        "display": "Black Eagle",
        "wiki": "Black_eagle"
    },

    "blakit1": {
        "display": "Black Kite",
        "wiki": "Black_kite"
    },

    "blhori1": {
        "display": "Black-hooded Oriole",
        "wiki": "Black-hooded_oriole"
    },

    "blnmon1": {
        "display": "Black-naped Monarch",
        "wiki": "Black-naped_monarch"
    },

    "blrwar1": {
        "display": "Black-throated Warbler",
        "wiki": "Black-throated_green_warbler"
    },

    "bncwoo3": {
        "display": "Brown-capped Woodpecker",
        "wiki": "Brown-capped_pygmy_woodpecker"
    },

    "brakit1": {
        "display": "Brahminy Kite",
        "wiki": "Brahminy_kite"
    },

    "brasta1": {
        "display": "Brahminy Starling",
        "wiki": "Brahminy_starling"
    },

    "brcful1": {
        "display": "Brown-cheeked Fulvetta",
        "wiki": "Brown-cheeked_fulvetta"
    },

    "brfowl1": {
        "display": "Brown Fish Owl",
        "wiki": "Brown_fish_owl"
    },

    "brnhao1": {
        "display": "Brown Hawk Owl",
        "wiki": "Brown_hawk-owl"
    },

    "brnshr": {
        "display": "Brown Shrike",
        "wiki": "Brown_shrike"
    },

    "brodro1": {
        "display": "Bronzed Drongo",
        "wiki": "Bronzed_drongo"
    },

    "brwjac1": {
        "display": "Bronze-winged Jacana",
        "wiki": "Bronze-winged_jacana"
    },

    "brwowl1": {
        "display": "Brown Wood Owl",
        "wiki": "Brown_wood_owl"
    },

    "btbeat1": {
        "display": "Blue-throated Bee-eater",
        "wiki": "Blue-throated_bee-eater"
    },

    "bwfshr1": {
        "display": "Black-winged Flycatcher-shrike",
        "wiki": "Black-winged_flycatcher-shrike"
    },

    "categr": {
        "display": "Cattle Egret",
        "wiki": "Cattle_egret"
    },

    "chbeat1": {
        "display": "Chestnut-headed Bee-eater",
        "wiki": "Chestnut-headed_bee-eater"
    },

    "cohcuc1": {
        "display": "Common Hawk-Cuckoo",
        "wiki": "Common_hawk-cuckoo"
    },

    "comfla1": {
        "display": "Common Flameback",
        "wiki": "Black-rumped_flameback"
    },

    "comgre": {
        "display": "Common Greenshank",
        "wiki": "Common_greenshank"
    },

    "comior1": {
        "display": "Common Iora",
        "wiki": "Common_iora"
    },

    "comkin1": {
        "display": "Common Kingfisher",
        "wiki": "Common_kingfisher"
    },

    "commoo3": {
        "display": "Common Moorhen",
        "wiki": "Common_moorhen"
    },

    "commyn": {
        "display": "Common Myna",
        "wiki": "Common_myna"
    },

    "compea": {
        "display": "Indian Peafowl",
        "wiki": "Indian_peafowl"
    },

    "comros": {
        "display": "Common Rosefinch",
        "wiki": "Common_rosefinch"
    },

    "comsan": {
        "display": "Common Sandpiper",
        "wiki": "Common_sandpiper"
    },

    "comtai1": {
        "display": "Common Tailorbird",
        "wiki": "Common_tailorbird"
    },

    "copbar1": {
        "display": "Coppersmith Barbet",
        "wiki": "Coppersmith_barbet"
    },

    "crbsun2": {
        "display": "Crimson-backed Sunbird",
        "wiki": "Crimson-backed_sunbird"
    },

    "cregos1": {
        "display": "Crested Goshawk",
        "wiki": "Crested_goshawk"
    },

    "crfbar1": {
        "display": "Crested Finchbill",
        "wiki": "Crested_finchbill"
    },

    "crseag1": {
        "display": "Crested Serpent Eagle",
        "wiki": "Crested_serpent_eagle"
    },

    "dafbab1": {
        "display": "Dark-fronted Babbler",
        "wiki": "Dark-fronted_babbler"
    },

    "darter2": {
        "display": "Oriental Darter",
        "wiki": "Oriental_darter"
    },

    "eaywag1": {
        "display": "Eastern Yellow Wagtail",
        "wiki": "Eastern_yellow_wagtail"
    },

    "emedov2": {
        "display": "Emerald Dove",
        "wiki": "Common_emerald_dove"
    },

    "eucdov": {
        "display": "Eurasian Collared Dove",
        "wiki": "Eurasian_collared_dove"
    },

    "eurbla2": {
        "display": "Eurasian Blackbird",
        "wiki": "Common_blackbird"
    },

    "eurcoo": {
        "display": "Eurasian Coot",
        "wiki": "Eurasian_coot"
    },

    "forwag1": {
        "display": "Forest Wagtail",
        "wiki": "Forest_wagtail"
    },

    "gargan": {
        "display": "Garganey",
        "wiki": "Garganey"
    },

    "gloibi": {
        "display": "Glossy Ibis",
        "wiki": "Glossy_ibis"
    },

    "goflea1": {
        "display": "Golden-fronted Leafbird",
        "wiki": "Golden-fronted_leafbird"
    },

    "graher1": {
        "display": "Grey Heron",
        "wiki": "Grey_heron"
    },

    "grbeat1": {
        "display": "Green Bee-eater",
        "wiki": "Green_bee-eater"
    },

    "grecou1": {
        "display": "Greater Coucal",
        "wiki": "Greater_coucal"
    },

    "greegr": {
        "display": "Green Bee-eater",
        "wiki": "Green_bee-eater"
    },

    "grefla1": {
        "display": "Greater Flameback",
        "wiki": "Greater_flameback"
    },

    "grehor1": {
        "display": "Grey Hornbill",
        "wiki": "Indian_grey_hornbill"
    },

    "grejun2": {
        "display": "Grey Junglefowl",
        "wiki": "Grey_junglefowl"
    },

    "grenig1": {
        "display": "Greenish Warbler",
        "wiki": "Greenish_warbler"
    },

    "grewar3": {
        "display": "Green Warbler",
        "wiki": "Green_warbler"
    },

    "grnsan": {
        "display": "Green Sandpiper",
        "wiki": "Green_sandpiper"
    },

    "grnwar1": {
        "display": "Green Warbler",
        "wiki": "Green_warbler"
    },

    "grtdro1": {
        "display": "Greater Racket-tailed Drongo",
        "wiki": "Greater_racket-tailed_drongo"
    },

        "gryfra": {
        "display": "Grey Francolin",
        "wiki": "Grey_francolin"
    },

    "grynig2": {
        "display": "Grey-headed Canary-flycatcher",
        "wiki": "Grey-headed_canary-flycatcher"
    },

    "grywag": {
        "display": "Grey Wagtail",
        "wiki": "Grey_wagtail"
    },

    "gybpri1": {
        "display": "Grey-breasted Prinia",
        "wiki": "Grey-breasted_prinia"
    },

    "gyhcaf1": {
        "display": "Grey-headed Canary Flycatcher",
        "wiki": "Grey-headed_canary-flycatcher"
    },

    "heswoo1": {
        "display": "Heart-spotted Woodpecker",
        "wiki": "Heart-spotted_woodpecker"
    },

    "hoopoe": {
        "display": "Eurasian Hoopoe",
        "wiki": "Hoopoe"
    },

    "houcro1": {
        "display": "House Crow",
        "wiki": "House_crow"
    },

    "houspa": {
        "display": "House Sparrow",
        "wiki": "House_sparrow"
    },

    "inbrob1": {
        "display": "Indian Robin",
        "wiki": "Indian_robin"
    },

    "indpit1": {
        "display": "Indian Pitta",
        "wiki": "Indian_pitta"
    },

    "indrob1": {
        "display": "Indian Robin",
        "wiki": "Indian_robin"
    },

    "indrol2": {
        "display": "Indian Roller",
        "wiki": "Indian_roller"
    },

    "indtit1": {
        "display": "Indian Tit",
        "wiki": "Cinereous_tit"
    },

    "ingori1": {
        "display": "Indian Golden Oriole",
        "wiki": "Indian_golden_oriole"
    },

    "inpher1": {
        "display": "Indian Pond Heron",
        "wiki": "Indian_pond_heron"
    },

    "insbab1": {
        "display": "Indian Scimitar Babbler",
        "wiki": "Indian_scimitar_babbler"
    },

    "insowl1": {
        "display": "Indian Scops Owl",
        "wiki": "Indian_scops_owl"
    },

    "integr": {
        "display": "Intermediate Egret",
        "wiki": "Intermediate_egret"
    },

    "isbduc1": {
        "display": "Indian Spot-billed Duck",
        "wiki": "Indian_spot-billed_duck"
    },

    "jerbus2": {
        "display": "Jerdon's Bushlark",
        "wiki": "Jerdon%27s_bushlark"
    },

    "junbab2": {
        "display": "Jungle Babbler",
        "wiki": "Jungle_babbler"
    },

    "junmyn1": {
        "display": "Jungle Myna",
        "wiki": "Jungle_myna"
    },

    "junowl1": {
        "display": "Jungle Owlet",
        "wiki": "Jungle_owlet"
    },

    "kenplo1": {
        "display": "Kentish Plover",
        "wiki": "Kentish_plover"
    },

    "kerlau2": {
        "display": "Kerala Laughingthrush",
        "wiki": "Kerala_laughingthrush"
    },

    "labcro1": {
        "display": "Large-billed Crow",
        "wiki": "Large-billed_crow"
    },

    "laudov1": {
        "display": "Laughing Dove",
        "wiki": "Laughing_dove"
    },

    "lblwar1": {
        "display": "Large-billed Leaf Warbler",
        "wiki": "Large-billed_leaf_warbler"
    },

    "lesyel1": {
        "display": "Lesser Yellowlegs",
        "wiki": "Lesser_yellowlegs"
    },

    "lewduc1": {
        "display": "Lesser Whistling Duck",
        "wiki": "Lesser_whistling_duck"
    },

    "lirplo": {
        "display": "Little Ringed Plover",
        "wiki": "Little_ringed_plover"
    },

    "litegr": {
        "display": "Little Egret",
        "wiki": "Little_egret"
    },

    "litgre1": {
        "display": "Little Grebe",
        "wiki": "Little_grebe"
    },

    "litspi1": {
        "display": "Little Spiderhunter",
        "wiki": "Little_spiderhunter"
    },

    "litswi1": {
        "display": "Little Swift",
        "wiki": "Little_swift"
    },

    "lobsun2": {
        "display": "Loten's Sunbird",
        "wiki": "Loten%27s_sunbird"
    },

    "maghor2": {
        "display": "Malabar Grey Hornbill",
        "wiki": "Malabar_grey_hornbill"
    },

    "malpar1": {
        "display": "Malabar Parakeet",
        "wiki": "Malabar_parakeet"
    },

    "maltro1": {
        "display": "Malabar Trogon",
        "wiki": "Malabar_trogon"
    },

    "malwoo1": {
        "display": "Malabar Woodshrike",
        "wiki": "Malabar_woodshrike"
    },

    "marsan": {
        "display": "Marsh Sandpiper",
        "wiki": "Marsh_sandpiper"
    },

    "mawthr1": {
        "display": "Malabar Whistling Thrush",
        "wiki": "Malabar_whistling_thrush"
    },

    "moipig1": {
        "display": "Mountain Imperial Pigeon",
        "wiki": "Mountain_imperial_pigeon"
    },

    "nilfly2": {
        "display": "Nilgiri Flycatcher",
        "wiki": "Nilgiri_flycatcher"
    },

    "niwpig1": {
        "display": "Nicobar Pigeon",
        "wiki": "Nicobar_pigeon"
    },

    "nutman": {
        "display": "Nuthatch",
        "wiki": "Nuthatch"
    },

    "orihob2": {
        "display": "Oriental Hobby",
        "wiki": "Oriental_hobby"
    },

    "oripip1": {
        "display": "Oriental Pipit",
        "wiki": "Oriental_pipit"
    },

    "pabflo1": {
        "display": "Paddyfield Pipit",
        "wiki": "Paddyfield_pipit"
    },

    "paisto1": {
        "display": "Painted Stork",
        "wiki": "Painted_stork"
    },

    "piebus1": {
        "display": "Pied Bushchat",
        "wiki": "Pied_bush_chat"
    },

    "piekin1": {
        "display": "Pied Kingfisher",
        "wiki": "Pied_kingfisher"
    },

    "placuc3": {
        "display": "Plaintive Cuckoo",
        "wiki": "Plaintive_cuckoo"
    },

    "plaflo1": {
        "display": "Plain Flowerpecker",
        "wiki": "Plain_flowerpecker"
    },

    "plapri1": {
        "display": "Plain Prinia",
        "wiki": "Plain_prinia"
    },

    "plhpar1": {
        "display": "Plum-headed Parakeet",
        "wiki": "Plum-headed_parakeet"
    },

    "pomgrp2": {
        "display": "Pompadour Green Pigeon",
        "wiki": "Pompadour_green_pigeon"
    },

    "purher1": {
        "display": "Purple Heron",
        "wiki": "Purple_heron"
    },

    "pursun3": {
        "display": "Purple Sunbird",
        "wiki": "Purple_sunbird"
    },

    "pursun4": {
        "display": "Purple-rumped Sunbird",
        "wiki": "Purple-rumped_sunbird"
    },

    "purswa3": {
        "display": "Purple Swamphen",
        "wiki": "Purple_swamphen"
    },

    "putbab1": {
        "display": "Puff-throated Babbler",
        "wiki": "Puff-throated_babbler"
    },

    "redspu1": {
        "display": "Red Spurfowl",
        "wiki": "Red_spurfowl"
    },

    "rerswa1": {
        "display": "Red-rumped Swallow",
        "wiki": "Red-rumped_swallow"
    },

    "revbul": {
        "display": "Red-vented Bulbul",
        "wiki": "Red-vented_bulbul"
    },

    "rewbul": {
        "display": "Red-whiskered Bulbul",
        "wiki": "Red-whiskered_bulbul"
    },

    "rewlap1": {
        "display": "Red-wattled Lapwing",
        "wiki": "Red-wattled_lapwing"
    },

    "rocpig": {
        "display": "Rock Pigeon",
        "wiki": "Rock_dove"
    },

    "rorpar": {
        "display": "Rose-ringed Parakeet",
        "wiki": "Rose-ringed_parakeet"
    },

    "rossta2": {
        "display": "Rosy Starling",
        "wiki": "Rosy_starling"
    },

    "rufbab3": {
        "display": "Rufous Babbler",
        "wiki": "Rufous_babbler"
    },

    "ruftre2": {
        "display": "Rufous Treepie",
        "wiki": "Rufous_treepie"
    },

        "rufwoo2": {
        "display": "Rufous Woodpecker",
        "wiki": "Rufous_woodpecker"
    },

    "rutfly6": {
        "display": "Rusty-tailed Flycatcher",
        "wiki": "Rusty-tailed_flycatcher"
    },

    "sbeowl1": {
        "display": "Spotted Owlet",
        "wiki": "Spotted_owlet"
    },

    "scamin3": {
        "display": "Scaly-breasted Munia",
        "wiki": "Scaly-breasted_munia"
    },

    "shikra1": {
        "display": "Shikra",
        "wiki": "Shikra"
    },

    "smamin1": {
        "display": "Small Minivet",
        "wiki": "Small_minivet"
    },

    "sohmyn1": {
        "display": "Southern Hill Myna",
        "wiki": "Southern_hill_myna"
    },

    "spepic1": {
        "display": "Speckled Piculet",
        "wiki": "Speckled_piculet"
    },

    "spodov": {
        "display": "Spotted Dove",
        "wiki": "Spotted_dove"
    },

    "spoowl1": {
        "display": "Spot-bellied Eagle Owl",
        "wiki": "Spot-bellied_eagle-owl"
    },

    "sqtbul1": {
        "display": "Square-tailed Bulbul",
        "wiki": "Square-tailed_bulbul"
    },

    "stbkin1": {
        "display": "Stork-billed Kingfisher",
        "wiki": "Stork-billed_kingfisher"
    },

    "sttwoo1": {
        "display": "Streak-throated Woodpecker",
        "wiki": "Streak-throated_woodpecker"
    },

    "thbwar1": {
        "display": "Thick-billed Warbler",
        "wiki": "Thick-billed_warbler"
    },

    "tibfly3": {
        "display": "Tickell's Blue Flycatcher",
        "wiki": "Tickell%27s_blue_flycatcher"
    },

    "tilwar1": {
        "display": "Tickell's Leaf Warbler",
        "wiki": "Tickell%27s_leaf_warbler"
    },

    "vefnut1": {
        "display": "Velvet-fronted Nuthatch",
        "wiki": "Velvet-fronted_nuthatch"
    },

    "vehpar1": {
        "display": "Vernal Hanging Parrot",
        "wiki": "Vernal_hanging_parrot"
    },

    "wbbfly1": {
        "display": "White-browed Bulbul",
        "wiki": "White-browed_bulbul"
    },

    "wemhar1": {
        "display": "Western Marsh Harrier",
        "wiki": "Western_marsh_harrier"
    },

    "whbbul2": {
        "display": "White-browed Bulbul",
        "wiki": "White-browed_bulbul"
    },

    "whbsho3": {
        "display": "White-browed Shama",
        "wiki": "White-rumped_shama"
    },

    "whbtre1": {
        "display": "White-browed Treepie",
        "wiki": "White-bellied_treepie"
    },

    "whbwag1": {
        "display": "White-browed Wagtail",
        "wiki": "White-browed_wagtail"
    },

    "whbwat1": {
        "display": "White-breasted Waterhen",
        "wiki": "White-breasted_waterhen"
    },

    "whbwoo2": {
        "display": "White-bellied Woodpecker",
        "wiki": "White-bellied_woodpecker"
    },

    "whcbar1": {
        "display": "White-cheeked Barbet",
        "wiki": "White-cheeked_barbet"
    },

    "whiter2": {
        "display": "White Ibis",
        "wiki": "American_white_ibis"
    },

    "whrmun": {
        "display": "White-rumped Munia",
        "wiki": "White-rumped_munia"
    },

    "whtkin2": {
        "display": "White-throated Kingfisher",
        "wiki": "White-throated_kingfisher"
    },

    "woosan": {
        "display": "Wood Sandpiper",
        "wiki": "Wood_sandpiper"
    },

    "wynlau1": {
        "display": "Wyatt's Lark",
        "wiki": "Wyatt%27s_lark"
    },

    "yebbab1": {
        "display": "Yellow-billed Babbler",
        "wiki": "Yellow-billed_babbler"
    },

    "yebbul3": {
        "display": "Yellow-browed Bulbul",
        "wiki": "Yellow-browed_bulbul"
    },

    "zitcis1": {
        "display": "Zitting Cisticola",
        "wiki": "Zitting_cisticola"
    }

}


# ============================================================
# FETCH BIRD INFO FROM WIKIPEDIA
# ============================================================

# ============================================================
# FETCH BIRD INFO FROM WIKIPEDIA
# ============================================================

def fetch_bird_info(bird_code):

    # =====================================================
    # GET BIRD DATA
    # =====================================================

    bird_data = bird_names.get(

        bird_code,

        {
            "display": bird_code,
            "wiki": bird_code
        }

    )

    display_name = bird_data["display"]

    wiki_name = bird_data["wiki"]

    # =====================================================
    # WIKIPEDIA URL
    # =====================================================

    wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{wiki_name}"

    print(wiki_url)

    try:

        response = requests.get(

            wiki_url,

            headers={
                "User-Agent": "BirdDetectorUltra/1.0"
            }

        )

        data = response.json()

        title = data.get(
            "title",
            display_name
        )

        description = data.get(
            "extract",
            "No information available."
        )

        image = data.get(
            "thumbnail",
            {}
        ).get(
            "source",
            None
        )

    except Exception as e:

        print(e)

        title = display_name

        description = "No information available."

        image = None

    return {

        "title": title,

        "description": description,

        "image": image

    }

# ============================================================
# FEATURE EXTRACTION
# ============================================================

N_MELS = 64
SPEC_WIDTH = 128

def extract_features(audio_source):

    # =====================================================
    # HANDLE STREAMLIT UPLOADS / MIC AUDIO
    # =====================================================

    if isinstance(audio_source, bytes):

        audio_file = BytesIO(audio_source)

    else:

        audio_file = audio_source

    # =====================================================
    # LOAD AUDIO
    # =====================================================

    audio, sr = librosa.load(

        audio_file,

        sr=None,

        mono=True,

        duration=5

    )

    # =====================================================
    # MEL SPECTROGRAM
    # =====================================================

    mel_spec = librosa.feature.melspectrogram(

        y=audio,

        sr=sr,

        n_mels=N_MELS

    )

    mel_spec = librosa.power_to_db(

        mel_spec,

        ref=np.max

    )

    # =====================================================
    # FIX WIDTH
    # =====================================================

    if mel_spec.shape[1] < SPEC_WIDTH:

        pad_width = SPEC_WIDTH - mel_spec.shape[1]

        mel_spec = np.pad(

            mel_spec,

            pad_width=((0,0),(0,pad_width)),

            mode='constant'

        )

    else:

        mel_spec = mel_spec[:, :SPEC_WIDTH]

    # =====================================================
    # NORMALIZE
    # =====================================================

    mel_spec = mel_spec / (

        np.max(np.abs(mel_spec)) + 1e-6

    )

    mel_spec = mel_spec[..., np.newaxis]

    mel_spec = np.expand_dims(

        mel_spec,

        axis=0

    )

    return mel_spec, audio, sr

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_bird(features):

    prediction = model.predict(
        features,
        verbose=0
    )[0]

    top_indices = prediction.argsort()[-5:][::-1]

    results = []

    for idx in top_indices:

        bird_name = encoder.inverse_transform([idx])[0]

        confidence = float(prediction[idx])

        results.append(
            (bird_name, confidence)
        )

    return results

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("""
# 🐦 Bird Detector Ultra
""")

st.sidebar.markdown("""
### 🌍 AI Features

✅ 182 Bird Species  
✅ Residual CNN  
✅ Attention Mechanisms  
✅ Spectrogram Intelligence  
✅ Deep Audio Learning  
✅ 81.5% Accuracy  
""")

st.sidebar.markdown("---")

st.sidebar.markdown("""
### ⚡ Model Stats

- Parameters: 1.47M
- Audio Duration: 5 sec
- Spectrograms: Mel Scale
- Architecture: Residual Attention CNN
""")

# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="main-title">
🐦 Bird Detector Ultra
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
Large-Scale Environmental Audio Intelligence System
</div>
""", unsafe_allow_html=True)

# ============================================================
# METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown("""
    <div class="metric-card">
        <h2>182</h2>
        <p>Bird Species</p>
    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown("""
    <div class="metric-card">
        <h2>81.5%</h2>
        <p>Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown("""
    <div class="metric-card">
        <h2>24K+</h2>
        <p>Samples</p>
    </div>
    """, unsafe_allow_html=True)

with col4:

    st.markdown("""
    <div class="metric-card">
        <h2>1.47M</h2>
        <p>Parameters</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "🎧 Upload Bird Audio",
    type=["wav", "mp3", "ogg"]
)

# ============================================================
# REAL-TIME MICROPHONE
# ============================================================

st.markdown("""
## 🎤 Real-Time Bird Listening
""")

audio_bytes = audio_recorder(

    text="🎙️ Click To Record",

    recording_color="#FF4B4B",

    neutral_color="#00F5A0",

    icon_name="microphone",

    icon_size="2x"

)

# ============================================================
# HANDLE MIC AUDIO
# ============================================================

if audio_bytes:

    st.session_state["mic_audio"] = audio_bytes

# ============================================================
# SHOW RECORDED AUDIO
# ============================================================

if "mic_audio" in st.session_state:

    st.success("✅ Recording Captured!")

    st.audio(st.session_state["mic_audio"])

    # ==============================================
    # MANUAL PREDICTION BUTTON
    # ==============================================

    if st.button("🧠 Predict Bird From Recording"):

        uploaded_file = BytesIO(

            st.session_state["mic_audio"]

        )

# ============================================================
# MAIN PREDICTION
# ============================================================

if uploaded_file is not None:


    with st.spinner("🧠 AI is listening to the forest..."):

        time.sleep(2)

        features, audio, sr = extract_features(
            uploaded_file
        )

        results = predict_bird(features)

    top_bird = results[0][0]

    top_conf = results[0][1]

    # =====================================================
    # MAIN RESULT
    # =====================================================

    st.markdown(f"""
    <div class="glass-card">

    <h1 style="text-align:center;">
    🐦 {bird_names.get(top_bird, {}).get("display", top_bird)}
    </h1>

    <h3 style="text-align:center;color:#00F5A0;">
    Confidence: {top_conf:.2%}
    </h3>

    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # LIVE BIRD INFORMATION
    # =====================================================

    info = fetch_bird_info(top_bird)

    st.markdown("""
    ## 🦜 Bird Information
    """)

    col1, col2 = st.columns([1,2])

    # =================================================
    # IMAGE
    # =================================================

    with col1:

        if info["image"]:

            st.image(
                info["image"],
                use_container_width=True
            )

    # =================================================
    # DETAILS
    # =================================================

    with col2:

        st.markdown(f"""
        <div class="glass-card">

        <h1>
        🐦 {info['title']}
        </h1>

        <hr>

        <p>
        {info['description']}
        </p>

        </div>
        """, unsafe_allow_html=True)
        # =====================================================
        # TOP 5 PREDICTIONS
        # =====================================================

        st.markdown("""
        ## 🔍 Top Predictions
        """)

        for bird, conf in results:

            st.markdown(f"""
            <div class="prediction-card">

            <h3>🐦 {bird_names.get(bird, {}).get("display", bird)}</h3>

            <p>Confidence: {conf:.2%}</p>

            </div>
            """, unsafe_allow_html=True)

            st.progress(conf)

        # =====================================================
        # CHART
        # =====================================================

        st.markdown("""
        ## 📊 Prediction Confidence
        """)

        chart_data = pd.DataFrame({

            "Bird": [x[0] for x in results],

            "Confidence": [x[1] for x in results]

        })

        st.bar_chart(
            chart_data.set_index("Bird")
        )

        # =====================================================
        # MEL SPECTROGRAM
        # =====================================================

        st.markdown("""
        ## 🎼 Mel Spectrogram
        """)

        fig, ax = plt.subplots(
            figsize=(12,5)
        )

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=64
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        img = librosa.display.specshow(
            mel_db,
            x_axis='time',
            y_axis='mel',
            sr=sr,
            ax=ax
        )

        plt.colorbar(img, ax=ax)

        st.pyplot(fig)

        # =====================================================
        # AUDIO WAVEFORM
        # =====================================================

        st.markdown("""
        ## 📈 Audio Waveform
        """)

        fig2, ax2 = plt.subplots(
            figsize=(12,3)
        )

        librosa.display.waveshow(
            audio,
            sr=sr,
            ax=ax2
        )

        st.pyplot(fig2)

# ============================================================
# FOOTER
# ============================================================

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<center>

<h3>🌿 Built with Deep Learning & Bioacoustics</h3>

<p>
Residual CNN • Attention Mechanisms • Spectrogram Intelligence
</p>

</center>
""", unsafe_allow_html=True)