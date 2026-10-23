"""
MediNova Advanced AI Clinical Copilot & Conversational Triage Engine
===================================================================
A hybrid dual-engine clinical conversational assistant engineered for:
1. FIRST PRIORITY: Evidence-based clinical triage, emergency red-flag safety,
   symptom analysis, laboratory biomarker explanations, and seamless routing
   to MediNova's 9 validated diagnostic screening suites.
2. UNIVERSAL VERSATILITY: Conversational fluency answering ANY user query
   (general questions, greetings, science, everyday topics, medical guidance)
   without dead-ends or unhelpful generic refusals.
3. HYBRID CLOUD / LOCAL RESILIENCE:
   - Primary: Google Gemini 3.8 Flash via the official `google-genai` SDK
     (with OpenAI GPT-4o-mini fallback).
   - Local Offline Brain: Comprehensive medical and conversational knowledge base
     guaranteeing high-utility responses even when no external API key is configured.
"""

import os
import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("medinova.clinical_copilot")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ==============================================================================
# 1. ACUTE CLINICAL EMERGENCY RED-FLAG PROTOCOLS (HIGHEST PRIORITY)
# ==============================================================================

EMERGENCY_TRIGGERS = [
    # Cardiac
    'crushing chest pain', 'severe chest pain', 'chest tightness', 'heart attack',
    'cardiac arrest', 'chest pressure radiating', 'pain radiating to left arm',
    'pain radiating to jaw', 'cold sweat with chest pain',
    # Respiratory
    'cannot breathe', "can't breathe", 'gasping for air', 'difficulty breathing',
    'severe shortness of breath', 'choking', 'stridor', 'blue lips', 'cyanosis',
    'oxygen below 90', 'oxygen below 92', 'spo2 below 90',
    # Neurological / Cerebrovascular
    'stroke', 'face drooping', 'facial droop', 'arm weakness', 'arm drift',
    'slurred speech', 'sudden numbness', 'sudden loss of vision', 'thunderclap headache',
    'worst headache of life', 'seizure', 'convulsion', 'unconscious', 'passed out',
    'fainting', 'syncope', 'unresponsive',
    # Hemorrhage / Acute Abdomen
    'vomiting blood', 'coughing blood', 'hematemesis', 'hemoptysis',
    'massive bleeding', 'black tarry stools', 'melena', 'severe acute abdominal pain',
    # Infectious / Anaphylaxis / Metabolic
    'anaphylaxis', 'swollen throat', 'airway swelling', 'blackwater',
    'rigors with confusion', 'diabetic coma', 'dka', 'fruity breath with vomiting',
    'suicide', 'kill myself', 'end my life', 'harm myself'
]

# ==============================================================================
# 2. MEDICAL EXPLAINERS FOR BIOMARKERS & TESTS
# ==============================================================================

MEDICAL_EXPLAINERS = {
    'oldpeak': {
        'term': 'ST Depression (Oldpeak)',
        'domain': 'Heart',
        'route': '/heart',
        'action_label': '🫀 Heart Screening',
        'explanation': (
            '**Oldpeak** measures ST-segment depression in millimeters induced by exercise relative to a resting ECG. '
            'It is a key indicator of **cardiac subendocardial ischemia** (inadequate oxygenated blood flow to ventricular heart muscle). '
            'In clinical stress testing, an Oldpeak $\\ge 1.5 - 2.0\\,\\text{mm}$ strongly correlates with coronary artery stenosis.'
        )
    },
    'thallium': {
        'term': 'Thallium-201 Scintigraphy (Stress Perfusion Scan)',
        'domain': 'Heart',
        'route': '/heart',
        'action_label': '🫀 Heart Screening',
        'explanation': (
            '**Thallium-201 Scintigraphy** evaluates myocardial blood perfusion using radioactive tracers. '
            'A *"Normal"* scan shows uniform radiotracer uptake throughout the myocardium. '
            'A *"Reversible Defect"* indicates viable heart tissue suffering from stress-induced ischemia. '
            'A *"Fixed Defect"* indicates scarred, non-viable myocardium from a prior myocardial infarction (heart attack).'
        )
    },
    'hba1c': {
        'term': 'Glycated Hemoglobin (HbA1c)',
        'domain': 'Diabetes',
        'route': '/diabetes',
        'action_label': '🩸 Diabetes Screening',
        'explanation': (
            '**HbA1c** measures the percentage of hemoglobin bound to glucose in the bloodstream, providing an accurate 90-to-120 day average of glycemic control. '
            '- Normal: $< 5.7\\%$\n'
            '- Prediabetes: $5.7\\% - 6.4\\%$\n'
            '- Type 2 Diabetes: $\\ge 6.5\\%$\n'
            'Every $1\\%$ reduction in HbA1c reduces microvascular complications (retinopathy, neuropathy, nephropathy) by approximately $37\\%$.'
        )
    },
    'creatinine': {
        'term': 'Serum Creatinine & Glomerular Filtration Rate (GFR)',
        'domain': 'Kidney',
        'route': '/kidney',
        'action_label': '🩺 Kidney Screening',
        'explanation': (
            '**Creatinine** is a chemical byproduct of normal muscle metabolism cleared exclusively by healthy renal glomeruli. '
            'Normal reference: $0.7 - 1.2\\,\\text{mg/dL}$ (men), $0.5 - 1.0\\,\\text{mg/dL}$ (women). '
            'Elevations above $1.3 - 1.5\\,\\text{mg/dL}$ indicate impaired glomerular filtration rate (GFR) and renal compromise.'
        )
    },
    'blood urea': {
        'term': 'Blood Urea & Blood Urea Nitrogen (BUN)',
        'domain': 'Kidney',
        'route': '/kidney',
        'action_label': '🩺 Kidney Screening',
        'explanation': (
            '**Blood Urea** is the primary nitrogenous waste product of protein metabolism synthesized in the liver and excreted by the kidneys. '
            'Normal range: $15 - 45\\,\\text{mg/dL}$. '
            'Elevated blood urea (azotemia) suggests diminished renal perfusion, dehydration, congestive heart failure, or acute kidney injury.'
        )
    },
    'albumin': {
        'term': 'Urine Albumin & Microalbuminuria',
        'domain': 'Kidney',
        'route': '/kidney',
        'action_label': '🩺 Kidney Screening',
        'explanation': (
            '**Albumin** is an essential blood plasma protein. Healthy glomerular filtration barriers prevent albumin from entering urine. '
            'Detection of urinary albumin (microalbuminuria or macroalbuminuria) is an early marker of diabetic nephropathy, hypertensive glomerulosclerosis, or intrinsic renal disease.'
        )
    },
    'bilirubin': {
        'term': 'Total & Direct Bilirubin',
        'domain': 'Liver',
        'route': '/liver',
        'action_label': '🧬 Liver Screening',
        'explanation': (
            '**Bilirubin** is a yellowish bile pigment produced during normal erythrocyte (red blood cell) catabolism. '
            'Normal total bilirubin: $0.2 - 1.2\\,\\text{mg/dL}$. '
            'Levels exceeding $2.0\\,\\text{mg/dL}$ cause clinical jaundice (icterus) with yellowing of the sclera and skin. '
            'Elevated direct (conjugated) bilirubin points to biliary obstruction or cholestasis; elevated indirect (unconjugated) bilirubin suggests hemolysis or hepatic clearance defects.'
        )
    },
    'alt': {
        'term': 'Alanine Aminotransferase (ALT / SGPT)',
        'domain': 'Liver',
        'route': '/liver',
        'action_label': '🧬 Liver Screening',
        'explanation': (
            '**ALT** is an intracellular transaminase enzyme concentrated primarily in liver hepatocytes. '
            'Normal reference: $7 - 56\\,\\text{U/L}$. '
            'When liver cells suffer inflammatory or toxic damage (such as in viral hepatitis, fatty liver, or drug toxicity), ALT leaks into the bloodstream, making it a sensitive liver-specific marker.'
        )
    },
    'ast': {
        'term': 'Aspartate Aminotransferase (AST / SGOT)',
        'domain': 'Liver',
        'route': '/liver',
        'action_label': '🧬 Liver Screening',
        'explanation': (
            '**AST** is an enzyme found in hepatocytes as well as heart and skeletal muscle. '
            'Normal reference: $10 - 40\\,\\text{U/L}$. '
            'An elevated AST/ALT ratio (De Ritis ratio $> 2.0$) frequently suggests alcoholic hepatitis, ischemic liver injury, or advanced cirrhosis.'
        )
    },
    'ag ratio': {
        'term': 'Albumin-to-Globulin (A/G) Ratio',
        'domain': 'Liver',
        'route': '/liver',
        'action_label': '🧬 Liver Screening',
        'explanation': (
            'The **A/G Ratio** compares albumin (synthesized exclusively by liver cells) against globulins (produced by immune cells). '
            'Normal reference: $1.0 - 2.2$. '
            'An inverted ratio ($< 1.0$) typically indicates impaired hepatic protein synthesis (cirrhosis) or excessive urinary protein loss (nephrotic syndrome).'
        )
    },
    'tsh': {
        'term': 'Thyroid-Stimulating Hormone (TSH)',
        'domain': 'Thyroid',
        'route': '/thyroid',
        'action_label': '🦋 Thyroid Screening',
        'explanation': (
            '**TSH** is released by the anterior pituitary gland to stimulate thyroid hormone production (T3 and T4). '
            'Normal clinical reference: $0.4 - 4.0\\,\\text{mIU/L}$. '
            'Elevated TSH ($> 4.5\\,\\text{mIU/L}$) indicates **Hypothyroidism** (underactive thyroid); low TSH ($< 0.4\\,\\text{mIU/L}$) indicates **Hyperthyroidism** (overactive thyroid).'
        )
    },
    'ecg': {
        'term': 'Electrocardiogram (ECG / EKG) & Rhythm Strip',
        'domain': 'Arrhythmia',
        'route': '/arrhythmia',
        'action_label': '⚡ ECG Arrhythmia',
        'explanation': (
            'An **ECG** records the electrical conduction system of the heart across cardiac cycles. '
            'Key components: **P-wave** (atrial depolarization), **QRS complex** (ventricular depolarization, normal width $< 120\\,\\text{ms}$), and **T-wave** (ventricular repolarization). '
            'MediNova processes Lead MLII at $360\\,\\text{Hz}$ using Pan-Tompkins QRS detection to identify premature ventricular contractions (PVCs) and arrhythmic ectopy.'
        )
    },
    'fna': {
        'term': 'Fine Needle Aspiration (FNA) Nuclear Morphometry',
        'domain': 'Breast Cancer',
        'route': '/breast_cancer',
        'action_label': '🎗️ Breast Oncology Screening',
        'explanation': (
            '**FNA Cytology** evaluates cell nucleus dimensions under high-magnification microscopy. '
            'Benign breast tissue displays regular, uniform nuclei (mean radius $< 14.0\\,\\mu\\text{m}$). '
            'Malignant neoplastic cells display enlarged chromatin volume, irregular nuclear borders (perimeter $> 90\\,\\mu\\text{m}$), and increased surface areas.'
        )
    }
}

# ==============================================================================
# 3. CLINICAL DOMAIN DEFINITIONS & 2-TURN INTERACTION GRAPH
# ==============================================================================

CLINICAL_DOMAINS: Dict[str, Dict[str, Any]] = {
    'heart': {
        'name': 'Cardiovascular Health & Chest Symptoms',
        'route': '/heart',
        'action_label': '🫀 Open Heart Screening',
        'insight_key': 'heart_disease',
        'keywords': [
            'chest pain', 'chest pressure', 'heart palpitations', 'left arm pain', 'angina',
            'palpitations', 'high bp', 'blood pressure', 'hypertension', 'shortness of breath on walking',
            'heart attack', 'cardiac', 'heart', 'bp', 'cholesterol', 'cardiovascular'
        ],
        'q1_text': "To help you handle this safely, how severe are your chest symptoms or blood pressure readings right now, and how long have you had them?",
        'q1_chips': ["🔴 Severe / Started suddenly", "🟡 Moderate / Past few days", "🟢 Mild / Just checking", "📊 High Blood Pressure"],
        'q2_text': "Are you experiencing shortness of breath, unexplained cold sweating, dizziness, nausea, or pain radiating to your left arm, neck, or jaw?",
        'q2_chips': ["Yes, I have those symptoms", "No, none of those", "Only mild chest tightness", "I already ran a MediNova test"],
        'emergency_advice': (
            "🚨 **CRITICAL FIRST-AID — CALL 112 / 911 / 108 IMMEDIATELY:**\n"
            "- **Sit Down & Rest:** Sit upright and rest immediately in a well-ventilated room. Do not walk or exert yourself.\n"
            "- **Loosen Tight Clothing:** Loosen collar, necktie, and waistband to reduce vascular resistance.\n"
            "- **Alert Someone Nearby:** Alert a family member or colleague right now. Do not drive yourself.\n"
            "- **Emergency Dispatch:** Call 112 / 911 / 108 right away and request an ambulance with cardiac life-support."
        ),
        'immediate_care': (
            "✅ **Immediate Action — How to Handle This Right Now:**\n"
            "1. **Rest in an Upright Posture:** Sit with your back supported. Take slow, regulated abdominal breaths.\n"
            "2. **Check Resting Vitals:** Measure your resting blood pressure and pulse rate with a home cuff. Write down the numbers.\n"
            "3. **Zero Stimulants:** Avoid all caffeine, energy drinks, nicotine, alcohol, and high-sodium meals.\n"
            "4. **Monitor Warning Signs:** If you notice sudden chest pressure radiating to your arm or jaw, seek emergency medical care immediately."
        )
    },
    'diabetes': {
        'name': 'Diabetes & Glycemic Regulation',
        'route': '/diabetes',
        'action_label': '🩸 Open Diabetes Screening',
        'insight_key': 'diabetes',
        'keywords': [
            'diabetes', 'sugar', 'blood sugar', 'glucose', 'hba1c', 'frequent thirst',
            'excessive urination', 'peeing a lot', 'polydipsia', 'polyuria', 'sweet smelling urine',
            'fruity breath', 'insulin', 'prediabetes', 'glycemic'
        ],
        'q1_text': "To help you handle your blood sugar safely, what are your current glucose levels or symptoms, and how long have you noticed these changes?",
        'q1_chips': ["🔴 Very High (>250 mg/dL) / Sudden", "🟡 Moderately High / Past few days", "🟢 Mild / Routine checking", "📉 Low Blood Sugar (<70 mg/dL)"],
        'q2_text': "Are you having extreme unquenchable thirst, frequent urination at night, sudden blurred vision, nausea, or fruity-smelling breath?",
        'q2_chips': ["Yes, I have those symptoms", "No, none of those", "Just thirst & fatigue", "I already ran a MediNova test"],
        'emergency_advice': (
            "🚨 **CRITICAL FIRST-AID — DKA / ACUTE HYPERGLYCEMIA ALERT:**\n"
            "- **Immediate Glucose & Ketone Test:** Test blood glucose and urine ketones if a home meter is accessible.\n"
            "- **Call Emergency (112 / 911 / 108):** If glucose is >250–300 mg/dL accompanied by nausea, vomiting, confusion, or fruity breath, seek emergency care to rule out Diabetic Ketoacidosis (DKA).\n"
            "- **Do Not Strenuously Exercise:** Strenuous exertion with high ketones accelerates dehydration and metabolic acidosis."
        ),
        'immediate_care': (
            "✅ **Immediate Action — How to Handle This Right Now:**\n"
            "1. **Hydrate with Plain Water:** Drink 2 to 3 glasses of clean room-temperature water to help your kidneys clear excess glucose.\n"
            "2. **Strictly Cut Simple Sugars:** Avoid fruit juices, sodas, sweets, refined flours, and high-glycemic carbohydrates.\n"
            "3. **Gentle Walking:** A light 15-minute flat walk after meals activates peripheral muscle GLUT-4 glucose transporters.\n"
            "4. **Log Biomarkers:** Record fasting and 2-hour post-prandial blood glucose along with recent HbA1c values."
        )
    },
    'thyroid': {
        'name': 'Thyroid & Endocrine Function',
        'route': '/thyroid',
        'action_label': '🦋 Open Thyroid Screening',
        'insight_key': 'thyroid',
        'keywords': [
            'thyroid', 'tsh', 't3', 't4', 'hypothyroid', 'hypothyroidism', 'hyperthyroid',
            'hyperthyroidism', 'goiter', 'neck swelling', 'cold intolerance', 'unexplained weight gain',
            'dry skin', 'hair loss', 'sluggish', 'hashimoto'
        ],
        'q1_text': "To help evaluate your thyroid health, what specific symptoms (like sudden weight changes, extreme cold intolerance, sluggishness, or neck fullness) are you experiencing?",
        'q1_chips': ["🔴 Extreme fatigue & weight gain", "🟡 High TSH lab result", "🟢 Mild tiredness / Hair thinning", "🦋 Neck swelling / Goiter"],
        'q2_text': "Are you currently taking levothyroxine or antithyroid medications, or experiencing severe constipation, dry brittle skin, or facial puffiness?",
        'q2_chips': ["Yes, on thyroid medicine", "No medications yet", "Have dry skin & puffiness", "I have my TSH / T3 lab numbers"],
        'emergency_advice': (
            "🚨 **CRITICAL FIRST-AID — MYXEDEMA / THYROID STORM WARNING:**\n"
            "- Extreme lethargy, low body temperature, and confusion (Myxedema coma) or severe rapid heart palpitations, high fever, and delirium (Thyroid storm) are life-threatening endocrine emergencies.\n"
            "- Call emergency services (**112 / 911 / 108**) immediately."
        ),
        'immediate_care': (
            "✅ **Immediate Action — How to Handle This Right Now:**\n"
            "1. **Standard Serum Thyroid Panel:** Test morning fasting TSH, Free T4, and Total T3.\n"
            "2. **Consistent Medication Timing:** If taking levothyroxine, take it on an empty stomach with plain water at least 30 to 60 minutes before breakfast.\n"
            "3. **Balanced Iodine & Selenium:** Ensure adequate dietary selenium (brazil nuts, sunflower seeds) and iodized salt in moderation.\n"
            "4. **Run MediNova AI Thyroid Analysis:** Test your 8 endocrine analytes on MediNova's calibrated classifier."
        )
    },
    'arrhythmia': {
        'name': 'Cardiac Arrhythmia & ECG Telemetry',
        'route': '/arrhythmia',
        'action_label': '⚡ Open ECG Arrhythmia Telemetry',
        'insight_key': 'arrhythmia',
        'keywords': [
            'arrhythmia', 'irregular heartbeat', 'skipped beats', 'fluttering in chest',
            'racing heart', 'tachycardia', 'bradycardia', 'pvc', 'premature beat',
            'ecg', 'ekg', 'lead ii', 'pulse erratic'
        ],
        'q1_text': "To safely assess your cardiac rhythm, how frequently do you feel skipped beats, rapid fluttering, or irregular heartbeats?",
        'q1_chips': ["🔴 Constant fluttering / Dizziness", "🟡 Occasional skipped beats", "🟢 Mild palpitations after coffee", "📊 Have digital ECG file"],
        'q2_text': "Do the irregular heartbeats occur alongside sudden lightheadedness, near-fainting (presyncope), shortness of breath, or chest pressure?",
        'q2_chips': ["Yes, dizzy with fluttering", "No, just skipped beats", "Only after exercise/stress", "I have an ECG strip"],
        'emergency_advice': (
            "🚨 **CRITICAL FIRST-AID — MALIGNANT ARRHYTHMIA ALERT:**\n"
            "- Rapid irregular heartbeats accompanied by syncope (fainting), severe dizziness, or chest pain require immediate emergency evaluation (112 / 911 / 108).\n"
            "- Sit or lie down immediately to prevent falls and injury."
        ),
        'immediate_care': (
            "✅ **Immediate Action — How to Handle This Right Now:**\n"
            "1. **Sit & Rest Quietly:** Sit upright or recline comfortably. Take calm, steady abdominal breaths.\n"
            "2. **Eliminate Arrhythmia Triggers:** Stop all caffeine, nicotine, energy drinks, and decongestant cold medicines.\n"
            "3. **Hydration & Electrolytes:** Drink water and ensure adequate dietary potassium and magnesium.\n"
            "4. **Upload Lead MLII ECG:** Upload your digitized rhythm strip to MediNova's Pan-Tompkins signal analyzer."
        )
    },
    'kidney': {
        'name': 'Kidney & Renal Health',
        'route': '/kidney',
        'action_label': '🩺 Open Kidney Screening',
        'insight_key': 'kidney_disease',
        'keywords': [
            'kidney', 'foamy urine', 'frothy urine', 'swollen ankles', 'swelling in feet',
            'edema', 'puffiness around eyes', 'flank pain', 'kidney pain', 'creatinine', 'blood urea', 'renal'
        ],
        'q1_text': "To assess your kidney health safely, how severe is the swelling or urine change, and when did you first notice it?",
        'q1_chips': ["🔴 Severe swelling / Sudden drop in urine", "🟡 Moderate swelling / Foamy urine", "🟢 Mild puffiness / Just checking", "🧪 Abnormal lab creatinine"],
        'q2_text': "Have you noticed persistent foamy/bubbly urine, cola-colored urine, swelling around your ankles/eyes, or lower back flank pain?",
        'q2_chips': ["Yes, swelling and foamy urine", "No, none of those", "Only foamy urine", "I already ran a MediNova test"],
        'emergency_advice': (
            "🚨 **CRITICAL FIRST-AID — ACUTE RENAL / FLUID OVERLOAD ALERT:**\n"
            "- A sudden severe decrease in urine output paired with breathing difficulty or rapid full-body edema suggests acute renal compromise or pulmonary fluid overload.\n"
            "- Seek immediate emergency medical assessment (112 / 911 / 108)."
        ),
        'immediate_care': (
            "✅ **Immediate Action — How to Handle This Right Now:**\n"
            "1. **Discontinue OTC NSAID Painkillers:** Completely avoid ibuprofen, naproxen, and diclofenac—they constrict renal blood flow.\n"
            "2. **Strict Low-Sodium Diet:** Keep sodium intake under 1,500 mg/day; avoid table salt and processed foods to reduce fluid retention.\n"
            "3. **Monitor Urine Output & Fluid Intake:** Record daily fluid intake and track urine color and volume.\n"
            "4. **Control Blood Pressure:** Keep resting blood pressure well below 130/80 mm Hg."
        )
    },
    'liver': {
        'name': 'Liver & Hepatic Function',
        'route': '/liver',
        'action_label': '🧬 Open Liver Screening',
        'insight_key': 'liver_disease',
        'keywords': [
            'liver', 'yellow eyes', 'yellow skin', 'jaundice', 'dark urine', 'pale stools',
            'right abdominal pain', 'bilirubin', 'alt', 'ast', 'hepatic', 'cirrhosis', 'fatty liver'
        ],
        'q1_text': "To evaluate your liver status safely, how noticeable is the jaundice, abdominal pain, or fatigue, and how quickly did it develop?",
        'q1_chips': ["🔴 Sudden deep yellowing / Upper pain", "🟡 Noticeable jaundice / Past few days", "🟢 Mild fatigue / Routine check", "🧪 Elevated SGOT/SGPT enzymes"],
        'q2_text': "Have you observed yellowing in the whites of your eyes, tea-colored dark urine, clay-colored pale stools, or right upper abdominal swelling?",
        'q2_chips': ["Yes, yellow eyes and dark urine", "No, none of those", "Only fatigue & abdominal ache", "I already ran a MediNova test"],
        'emergency_advice': (
            "🚨 **CRITICAL FIRST-AID — ACUTE HEPATIC ENCEPHALOPATHY ALERT:**\n"
            "- Acute jaundice accompanied by mental confusion, severe drowsiness, slurred speech, or persistent vomiting requires immediate emergency hospitalization."
        ),
        'immediate_care': (
            "✅ **Immediate Action — How to Handle This Right Now:**\n"
            "1. **Complete Alcohol Cessation:** Eliminate 100% of alcohol intake immediately to protect hepatocytes from toxic stress.\n"
            "2. **Avoid Hepatotoxic Drugs:** Avoid paracetamol in high doses and discontinue unverified herbal supplements.\n"
            "3. **Clean Low-Fat Nutrition:** Eat light, easily digestible meals (boiled vegetables, lean chicken, lentils) and stay hydrated.\n"
            "4. **Complete Liver Panel (LFT):** Test Total Bilirubin, Direct Bilirubin, ALT, AST, ALP, and Albumin/Globulin ratio."
        )
    },
    'breast_cancer': {
        'name': 'Breast Oncology & Tissue Screening',
        'route': '/breast_cancer',
        'action_label': '🔬 Open Breast Oncology Screening',
        'insight_key': 'breast_cancer',
        'keywords': [
            'breast lump', 'lump in breast', 'nipple discharge', 'breast skin dimpling',
            'inverted nipple', 'peau d orange', 'breast pain', 'breast mass', 'breast cancer', 'mammogram'
        ],
        'q1_text': "To guide your clinical evaluation safely, how long have you noticed this breast lump or tissue change, and does it feel tender or hard?",
        'q1_chips': ["🔴 Hard / Fixed / Rapidly growing", "🟡 New palpable lump / Past weeks", "🟢 Cyclical tenderness / Hormonal", "🔬 Have biopsy/mammogram numbers"],
        'q2_text': "Have you observed any skin dimpling, spontaneous bloody/clear nipple discharge, recent nipple retraction, or swollen armpit lymph nodes?",
        'q2_chips': ["Yes, dimpling or discharge", "No, just a smooth lump", "No skin changes", "I already ran a MediNova test"],
        'emergency_advice': (
            "🚨 **CLINICAL ONCOLOGY ADVISORY:**\n"
            "- A hard, non-tender, fixed breast mass accompanied by skin dimpling or bloody nipple discharge requires priority evaluation by a surgical oncologist.\n"
            "- Schedule an in-person clinical breast exam, diagnostic mammogram, and ultrasound."
        ),
        'immediate_care': (
            "✅ **Immediate Action — How to Handle This Right Now:**\n"
            "1. **Do Not Repeatedly Squeeze or Manipulate:** Avoid aggressive palpation, which can induce focal inflammation.\n"
            "2. **Document Characteristics:** Note whether the lump changes size across your menstrual cycle.\n"
            "3. **Arrange Diagnostic Imaging:** Schedule a high-resolution bilateral mammogram and targeted ultrasound.\n"
            "4. **Run MediNova Cellular Morphometry:** Input cell nucleus parameters from your Fine Needle Aspiration (FNA) report."
        )
    },
    'malaria': {
        'name': 'Malaria Blood Smear Parasitology',
        'route': '/malaria',
        'action_label': '🦟 Open Malaria Screening',
        'insight_key': 'malaria',
        'keywords': [
            'malaria', 'shaking chills', 'rigors', 'cyclic fever', 'mosquito bite',
            'high spiking fever', 'profuse sweating', 'plasmodium', 'blood smear'
        ],
        'q1_text': "To assess suspected malaria, how high is your fever, does it cycle with intense shaking chills and sweating, and did you travel recently?",
        'q1_chips': ["🔴 High spiking fever (>103°F) + Chills", "🟡 Moderate fever / Past 2 days", "🟢 Mild temperature / Flu-like", "🦟 Recent mosquito exposure"],
        'q2_text': "Are you experiencing severe headache, vomiting, dark tea-colored urine, jaundice, or confusion along with the fevers?",
        'q2_chips': ["Yes, chills and tea-colored urine", "No, just fever & chills", "Mild body ache only", "I have a blood smear slide photo"],
        'emergency_advice': (
            "🚨 **CRITICAL FIRST-AID — CEREBRAL MALARIA & BLACKWATER ALERT:**\n"
            "- Spiking fevers with confusion, convulsions, or dark black tea-colored urine suggest severe Falciparum malaria with hemolysis.\n"
            "- This requires immediate emergency hospitalization and intravenous antimalarials."
        ),
        'immediate_care': (
            "✅ **Immediate Action — How to Handle This Right Now:**\n"
            "1. **Confirm via Peripheral Blood Smear:** Obtain a microscopic thick and thin Giemsa-stained blood film or rapid diagnostic test (RDT).\n"
            "2. **Continuous Oral Rehydration:** Drink oral rehydration salts (ORS), tender coconut water, and clean water.\n"
            "3. **Antipyretic Fever Control:** Use paracetamol according to physician dosage; avoid aspirin.\n"
            "4. **AI Micrograph Scan:** Upload your thin blood smear digital micrograph to MediNova's deep learning CNN."
        )
    },
    'pneumonia': {
        'name': 'Pulmonary & Pneumonia Radiography',
        'route': '/pneumonia',
        'action_label': '🫁 Open Pneumonia CXR Screening',
        'insight_key': 'pneumonia',
        'keywords': [
            'pneumonia', 'cough with phlegm', 'chest infection', 'crackling breathing',
            'pleuritic chest pain', 'coughing yellow green mucus', 'lung infection', 'chest x-ray', 'cxr'
        ],
        'q1_text': "To evaluate your respiratory symptoms, how long have you had this cough, what color is the mucus, and do you feel breathless?",
        'q1_chips': ["🔴 Severe breathlessness + High fever", "🟡 Productive green cough / Past week", "🟢 Mild dry cough / Scratchy throat", "🩻 Have chest X-ray image"],
        'q2_text': "Do you have sharp stabbing chest pain when breathing deeply, blueish lips or fingertips, or an oxygen saturation (SpO2) reading below 94%?",
        'q2_chips': ["Yes, sharp pain & low oxygen", "No, just chest congestion", "SpO2 is 96%+", "I have a CXR image to upload"],
        'emergency_advice': (
            "🚨 **CRITICAL FIRST-AID — ACUTE RESPIRATORY DISTRESS ALERT:**\n"
            "- Severe shortness of breath, oxygen saturation $< 92\\%$, cyanosis, or confusion requires immediate emergency room oxygen therapy (112 / 911 / 108)."
        ),
        'immediate_care': (
            "✅ **Immediate Action — How to Handle This Right Now:**\n"
            "1. **Continuous Pulse Oximetry:** Monitor oxygen saturation (SpO2) every 2 hours.\n"
            "2. **Rest Upright:** Elevate head and torso with 2–3 pillows to maximize lung alveolar recruitment.\n"
            "3. **Steam & Warm Hydration:** Inhale warm saline steam and drink warm soups and fluids to loosen secretions.\n"
            "4. **AI Chest X-Ray Analysis:** Upload an anterior-posterior chest radiograph to MediNova's deep learning model."
        )
    },
    'general': {
        'name': 'General Health & Clinical Consultation',
        'route': '/features',
        'action_label': '🔍 Explore All 9 Diagnostic Suites',
        'insight_key': 'general',
        'keywords': ['health', 'symptom', 'doctor', 'checkup', 'test', 'feeling sick', 'consult'],
        'q1_text': "How are you feeling overall right now, and which bodily area or symptom concerns you most?",
        'q1_chips': ["🫀 Heart / Blood Pressure", "🩸 Blood Sugar / Thirst", "🦋 Thyroid / Fatigue", "⚡ Heart Rhythm / Palpitations"],
        'q2_text': "Are any of these symptoms severe or sudden, or are you looking for preventive health guidance and screening?",
        'q2_chips': ["Severe / Sudden", "Routine preventive check", "Lab result review", "General question"],
        'emergency_advice': (
            "🚨 **EMERGENCY NOTICE:** For acute severe chest pain, shortness of breath, stroke signs, or severe injuries, call **112 / 911 / 108** immediately."
        ),
        'immediate_care': (
            "✅ **General Wellness Guidance:** Rest, monitor vitals (BP, pulse, temperature), hydrate with clean water, and select the appropriate MediNova diagnostic suite below."
        )
    }
}

# ==============================================================================
# 4. EXTENDED MEDICAL SYMPTOMS & PRIMARY CARE KNOWLEDGE BASE
# ==============================================================================

EXTENDED_SYMPTOMS_KB = {
    'headache': {
        'keywords': ['headache', 'head pain', 'migraine', 'throbbing head', 'temple pain', 'tension headache'],
        'title': 'Headache & Migraine Assessment',
        'overview': (
            'Headaches are among the most common clinical presentations. Most headaches are primary (Tension, Migraine, or Cluster), '
            'but it is critical to rule out secondary causes (vascular, hypertensive, or intracranial).'
        ),
        'red_flags': [
            'Sudden severe "thunderclap" headache (reaching peak intensity within seconds)',
            'Headache accompanied by high fever, stiff neck, and light sensitivity (meningitis)',
            'New-onset headache with focal neurological signs (slurred speech, vision loss, limb weakness)',
            'Headache following recent head trauma or in patients $> 50$ years with scalp tenderness'
        ],
        'home_care': [
            'Rest in a dark, quiet, temperature-controlled room.',
            'Hydrate with 500 mL of water (dehydration is a frequent migraine trigger).',
            'Apply a cold gel pack to the forehead or warm compress to the back of the neck.',
            'Gentle temple massage and slow diaphragmatic breathing.',
            'Track your headache triggers (stress, caffeine withdrawal, lack of sleep, screen time).'
        ],
        'action_route': '/contact',
        'action_label': '🏥 Consult a Neurologist / Primary Care'
    },
    'fever': {
        'keywords': ['fever', 'high temp', 'temperature', 'pyrexia', 'feverish', 'chills and body ache'],
        'title': 'Fever & Infectious Triage',
        'overview': (
            'A fever is an elevation of body temperature above the normal circadian range ($> 38.0^\\circ\\text{C}$ or $> 100.4^\\circ\\text{F}$), '
            'typically indicating an active immune response to viral, bacterial, or parasitic pathogens.'
        ),
        'red_flags': [
            'Fever $> 103^\\circ\\text{F}$ ($39.5^\\circ\\text{C}$) or persistent $> 3$ consecutive days without improvement',
            'Fever with difficulty breathing, cyanosis, or chest pain (Rule out Pneumonia)',
            'Spiking cyclic fever with severe shaking rigors (Rule out Malaria)',
            'Fever with a non-blanching purple petechial rash or altered mental confusion (Emergency)'
        ],
        'home_care': [
            'Continuous oral hydration: Drink water, oral rehydration solutions (ORS), and warm broths.',
            'Light clothing and lukewarm (not cold) sponging to reduce core temperature safely.',
            'Paracetamol for fever relief under medical dosage guidance; avoid aspirin in adolescents.',
            'Rest completely to allow immune cell proliferation and metabolic recovery.'
        ],
        'action_route': '/malaria',
        'action_label': '🦟 Screen for Malaria / Pneumonia'
    },
    'stomach_pain': {
        'keywords': ['stomach pain', 'stomach ache', 'abdominal pain', 'belly ache', 'cramps', 'gastric', 'acid reflux', 'gerd', 'indigestion', 'heartburn'],
        'title': 'Abdominal & Gastrointestinal Evaluation',
        'overview': (
            'Abdominal symptoms range from benign functional dyspepsia, acid reflux (GERD), and viral gastroenteritis '
            'to acute surgical conditions (appendicitis, cholecystitis, or pancreatitis).'
        ),
        'red_flags': [
            'Severe, sharp pain localizing to the right lower abdomen (potential appendicitis)',
            'Severe pain radiating to the back with vomiting (potential acute pancreatitis)',
            'Vomiting blood (hematemesis) or passing dark black tarry stools (gastrointestinal hemorrhage)',
            'Rigid "board-like" abdominal wall or high fever with intense abdominal rebound tenderness'
        ],
        'home_care': [
            'Bland diet: follow the BRAT protocol (Bananas, Rice, Applesauce, Toast).',
            'Avoid irritating triggers: zero alcohol, zero chili/spices, carbonated sodas, or fried foods.',
            'Small, frequent meals; remain upright for at least 2 hours after eating to curb GERD.',
            'Sip warm peppermint, chamomile, or ginger tea for mild spasmodic discomfort.'
        ],
        'action_route': '/liver',
        'action_label': '🧬 Screen Liver & Digestive Health'
    },
    'back_pain': {
        'keywords': ['back pain', 'lower back pain', 'lumbar pain', 'spine pain', 'sciatica', 'slipped disc', 'backache'],
        'title': 'Musculoskeletal & Lumbar Spine Assessment',
        'overview': (
            'Lower back pain is predominantly mechanical (muscular strain, lumbar facet arthrosis, or disc herniation). '
            'Conservative rehabilitation is effective for $> 90\\%$ of cases within 4 to 6 weeks.'
        ),
        'red_flags': [
            'Cauda Equina Syndrome: Sudden loss of bowel or bladder control, or progressive saddle numbness',
            'Progressive foot drop or severe bilateral lower limb weakness',
            'Unexplained fever, night sweats, or history of cancer accompanied by severe spinal pain',
            'Intense unremitting back pain following major physical trauma or fall'
        ],
        'home_care': [
            'Short-term relative rest: Avoid prolonged bed rest beyond 24–48 hours; gentle walking aids recovery.',
            'Ice pack therapy for the first 48 hours (15–20 min intervals), followed by soothing moist heat.',
            'Ergonomic posture: Maintain neutral lumbar curvature; avoid heavy twisting or bending.',
            'Core stability exercises (bird-dog, pelvic tilts) once acute spasm subsides.'
        ],
        'action_route': '/contact',
        'action_label': '🏥 Book Orthopedic / Spine Specialist'
    },
    'dizziness': {
        'keywords': ['dizziness', 'dizzy', 'vertigo', 'lightheaded', 'room spinning', 'unsteady', 'loss of balance'],
        'title': 'Dizziness & Vestibular Assessment',
        'overview': (
            'Dizziness encompasses vertigo (a false sensation of spinning, often vestibular), '
            'presyncope (feeling faint, often cardiovascular or orthostatic), and disequilibrium (balance impairment).'
        ),
        'red_flags': [
            'Dizziness with sudden facial weakness, slurred speech, or double vision (Stroke / CVA)',
            'Dizziness accompanied by chest pain, rapid fluttering palpitations, or shortness of breath',
            'Complete loss of consciousness (syncope) or profound inability to stand or sit upright'
        ],
        'home_care': [
            'Sit or lie down immediately in a well-ventilated space to prevent fall trauma.',
            'Drink 2 glasses of water (orthostatic hypotension is frequently provoked by dehydration).',
            'Avoid sudden postural changes: sit on the edge of the bed for 30 seconds before standing.',
            'Avoid driving or operating machinery until balance is fully and stably restored.'
        ],
        'action_route': '/arrhythmia',
        'action_label': '⚡ Check Heart Rhythm / Telemetry'
    },
    'fatigue': {
        'keywords': ['fatigue', 'tired', 'tiredness', 'exhausted', 'exhaustion', 'low energy', 'lethargic', 'sluggish all day'],
        'title': 'Chronic Fatigue & Metabolic Screening',
        'overview': (
            'Persistent fatigue requires systematic evaluation of sleep architecture, endocrine status, '
            'hematologic capacity, and metabolic organ function.'
        ),
        'red_flags': [
            'Fatigue accompanied by shortness of breath on minimal exertion or chest tightness',
            'Profound unexplained weight loss, night sweats, or chronic fever',
            'Pale conjunctiva, cold extremities, and dizziness (severe anemia)'
        ],
        'home_care': [
            'Check Thyroid Status: Hypothyroidism is a prime cause of unexplained exhaustion (TSH $> 4.0$).',
            'Check Blood Sugar: Chronic hyperglycemia or insulin resistance causes cellular energy starvation.',
            'Review Sleep Hygiene: Aim for 7–8 hours of consistent, uninterrupted sleep in darkness.',
            'Screen for Nutritional Deficiencies: Check Serum Ferritin, Vitamin D3, and Vitamin B12.'
        ],
        'action_route': '/thyroid',
        'action_label': '🦋 Screen Thyroid & Metabolic Panel'
    },
    'stress_anxiety': {
        'keywords': ['anxiety', 'stress', 'panic attack', 'nervous', 'worried', 'hyperventilating', 'cannot relax', 'heart racing anxiety'],
        'title': 'Stress, Anxiety & Parasympathetic Regulation',
        'overview': (
            'Acute anxiety triggers sympathetic nervous system fight-or-flight release (adrenaline and cortisol), '
            'causing tachycardia, hyperventilation, trembling, and chest tightness.'
        ),
        'red_flags': [
            'Chest pain that is crushing or radiates to jaw/arm (must rule out acute cardiac event first)',
            'Thoughts of self-harm, severe clinical depression, or acute psychosis (seek immediate support)'
        ],
        'home_care': [
            'Box Breathing Protocol: Inhale 4 seconds, hold 4 seconds, exhale 4 seconds, hold 4 seconds.',
            'Grounding Technique (5-4-3-2-1): Notice 5 things you see, 4 you feel, 3 you hear, 2 you smell, 1 you taste.',
            'Cut Stimulants: Eliminate caffeine, energy drinks, and excessive screen time.',
            'Physical De-escalation: Progressive muscle relaxation from toes to forehead.'
        ],
        'action_route': '/features',
        'action_label': '🧘 View Wellness & Triage Portal'
    }
}

# ==============================================================================
# 5. MEDICATIONS & PHARMACOLOGICAL SAFETY KNOWLEDGE BASE
# ==============================================================================

MEDICATIONS_KB = {
    'paracetamol': {
        'keywords': ['paracetamol', 'acetaminophen', 'tylenol', 'crocin', 'dolo', 'calpol'],
        'name': 'Paracetamol (Acetaminophen)',
        'class': 'Analgesic & Antipyretic',
        'indication': 'Mild to moderate pain relief and fever reduction.',
        'safe_dosage': 'Standard adult dosage is 500 mg to 650 mg every 4–6 hours as needed. Maximum 24-hour adult intake must strictly not exceed 3,000 to 4,000 mg.',
        'safety_warnings': [
            'Hepatotoxicity: Do not take with alcohol or multiple combination cold medications containing acetaminophen.',
            'Patients with underlying liver cirrhosis or elevated ALT/AST must consult their hepatologist before use.'
        ],
        'relevant_suite': '/liver'
    },
    'ibuprofen': {
        'keywords': ['ibuprofen', 'advil', 'motrin', 'brufen', 'nsaid', 'naproxen', 'diclofenac'],
        'name': 'Ibuprofen (NSAID)',
        'class': 'Non-Steroidal Anti-Inflammatory Drug',
        'indication': 'Inflammatory pain, musculoskeletal sprains, toothache, and menstrual cramps.',
        'safe_dosage': 'Standard adult dose is 200 mg to 400 mg every 6–8 hours with or after food. Maximum OTC daily limit is 1,200 mg.',
        'safety_warnings': [
            'Renal Risk: NSAIDs inhibit renal prostaglandins and can precipitate acute kidney injury. Avoid in chronic kidney disease.',
            'Gastric Risk: Always take with food to prevent gastritis or peptic ulceration.',
            'Cardiovascular Caution: Long-term NSAID use can elevate blood pressure and fluid retention.'
        ],
        'relevant_suite': '/kidney'
    },
    'metformin': {
        'keywords': ['metformin', 'glucophage', 'glycomet'],
        'name': 'Metformin',
        'class': 'Biguanide Antidiabetic Agent',
        'indication': 'First-line therapy for Type 2 Diabetes Mellitus; reduces hepatic gluconeogenesis.',
        'safe_dosage': 'Prescription medication. Typically initiated at 500 mg once or twice daily with main meals to reduce GI disturbance.',
        'safety_warnings': [
            'Take with meals to minimize nausea, bloating, or diarrhea.',
            'Monitor kidney function: Contraindicated if estimated GFR (eGFR) falls below 30 mL/min/1.73m².',
            'Periodic monitoring of Vitamin B12 levels is recommended with long-term use.'
        ],
        'relevant_suite': '/diabetes'
    },
    'statin': {
        'keywords': ['statin', 'atorvastatin', 'rosuvastatin', 'lipitor', 'crestor', 'cholesterol medicine'],
        'name': 'Statins (Atorvastatin / Rosuvastatin)',
        'class': 'HMG-CoA Reductase Inhibitor',
        'indication': 'Lowers LDL cholesterol and stabilizes atherosclerotic plaque in cardiovascular disease prevention.',
        'safe_dosage': 'Prescription medication taken once daily, typically in the evening.',
        'safety_warnings': [
            'Report unexplained muscle pain, tenderness, or weakness (rare risk of myopathy/rhabdomyolysis).',
            'Periodic monitoring of liver enzymes (ALT/AST) is recommended during long-term therapy.'
        ],
        'relevant_suite': '/heart'
    },
    'levothyroxine': {
        'keywords': ['levothyroxine', 'synthroid', 'eltroxin', 'thyronorm', 'thyroid pill'],
        'name': 'Levothyroxine (T4 Replacement)',
        'class': 'Synthetic Thyroid Hormone',
        'indication': 'Hormone replacement therapy for primary, secondary, and subclinical hypothyroidism.',
        'safe_dosage': 'Strictly individualized by physician based on morning fasting TSH levels.',
        'safety_warnings': [
            'Take strictly on an empty stomach with a full glass of plain water at least 30 to 60 minutes before breakfast.',
            'Do not take calcium supplements, iron pills, or soy products within 4 hours of levothyroxine (they block absorption).',
            'Retest TSH 6 to 8 weeks after any dose adjustment.'
        ],
        'relevant_suite': '/thyroid'
    },
    'aspirin': {
        'keywords': ['aspirin', 'ecotrin', 'disprin', 'low dose aspirin', 'blood thinner aspirin'],
        'name': 'Aspirin (Acetylsalicylic Acid)',
        'class': 'Antiplatelet & Analgesic',
        'indication': 'Secondary prevention of recurrent myocardial infarction and ischemic stroke (75–100 mg daily).',
        'safe_dosage': 'Low-dose cardioprotective aspirin is typically 75 mg to 100 mg once daily, strictly as prescribed.',
        'safety_warnings': [
            'Bleeding risk: Do not initiate without medical consultation, as it elevates major gastrointestinal and cerebral bleeding risk.',
            'Never give aspirin to children or teenagers with viral symptoms due to Reye syndrome risk.'
        ],
        'relevant_suite': '/heart'
    }
}

# ==============================================================================
# 6. GENERAL CONVERSATION & PLATFORM GUIDANCE KNOWLEDGE BASE
# ==============================================================================

GENERAL_CONVERSATION_RESPONSES = {
    'greetings': [
        "Hello! I am your **MediNova Clinical AI Copilot & Medical Advisor**.",
        "Hi there! Welcome to MediNova Clinical Decision Support. How can I assist you with your health, symptoms, or lab reports today?",
        "Greetings! I'm here to provide evidence-based clinical guidance, biomarker explanations, and risk screening across our 9 medical modules."
    ],
    'creator_info': (
        "### 👨‍💻 MediNova Architecture & Credits\n\n"
        "**MediNova** was architected and developed by **DHARSAN KUMAR R**.\n\n"
        "- **Platform:** Advanced AI-Powered Clinical Decision Support Suite (CDSS v5.0.0)\n"
        "- **Core Technologies:** Python Flask, Scikit-Learn, TensorFlow/Keras, XGBoost, Pan-Tompkins Signal Processing, OpenStreetMap Geolocation, and Cryptographic Report Verification\n"
        "- **Clinical Standards:** Governed under ISO/IEC 42001 and AAMI EC57 evaluation benchmarks.\n\n"
        "How can I help you navigate the system or analyze your health today?"
    ),
    'capabilities_overview': (
        "### 🌟 What MediNova AI Copilot Can Do For You\n\n"
        "**1. Clinical Decision Support & Symptom Triage (First Priority):**\n"
        "- Assess symptoms across 9 medical specialties: *Heart Disease, Diabetes, Kidney Disease, Liver Health, Breast Oncology, Thyroid Dysfunction, Cardiac Arrhythmia, Malaria, and Pneumonia*.\n"
        "- Issue instant alerts and guidance for acute emergency red flags.\n\n"
        "**2. Lab Report Scanner & Biomarker Explanations:**\n"
        "- Upload PDFs or photo scans of your lab reports for automatic biomarker extraction.\n"
        "- Explain clinical test values (HbA1c, TSH, Serum Creatinine, Bilirubin, ECG waves, etc.) with standard reference intervals.\n\n"
        "**3. Doctor Appointment Booking & Hospital Discovery:**\n"
        "- Locate accredited specialist departments within a 5–100 km radius via our 7-tier geocoder.\n"
        "- Schedule priority hospital consultations with digital calendar exports (.ics).\n\n"
        "**4. General Knowledge & Health Advice:**\n"
        "- Ask me any general questions about nutrition, exercise, pharmacology, science, or technology!"
    ),
    'appointment_guide': (
        "### 📅 How to Book a Priority Hospital Consultation\n\n"
        "1. Navigate to the **[Find Specialist Hospitals & Care Network](/contact)** page.\n"
        "2. Enter your address, district, or PIN code, or click **Use GPS Location**.\n"
        "3. Choose your medical condition; our engine filters strictly for verified departments (e.g. Pulmonology for Pneumonia, Cardiology for Heart Disease).\n"
        "4. Click **Book Priority Consultation** on your chosen facility.\n"
        "5. Enter your contact details and preferred time slot to generate your encrypted reference ID (`APT-YYYY-XXXXXX`) and digital calendar invite."
    ),
    'scanner_guide': (
        "### 📄 How to Use the Smart Lab Report Scanner\n\n"
        "1. Open any screening page or head to the **[Diagnostic Features Selector](/features#scanner)**.\n"
        "2. Click **📄 Scan Lab Report (PDF / Image)**.\n"
        "3. Select a digital lab report PDF or photo (PNG, JPG, WebP) up to 16 MB.\n"
        "4. MediNova extracts clinical parameters using optical text extraction and converts units automatically (e.g. $\\text{mmol/L} \\to \\text{mg/dL}$).\n"
        "5. Choose **Pre-fill Diagnostic Form** to review the numbers, or **Instant AI Triage** for an immediate risk assessment!"
    )
}

# ==============================================================================
# 7. MULTIMODAL CLOUD LLM CALLER (GEMINI 3.8 FLASH / OPENAI)
# ==============================================================================

def call_live_llm(user_msg: str, current_page: str, page_context: Dict[str, Any], last_diagnosis: Optional[Dict[str, Any]], chat_history: Optional[List[Dict[str, str]]] = None) -> Optional[Dict[str, Any]]:
    """
    Invokes Google Gemini 3.8 Flash via the google-genai SDK (or OpenAI GPT-4o-mini).
    Enforces user's primary requirement:
      1. FIRST PRIORITY: Evidence-based clinical assessment, symptom triage, emergency precautions,
         biomarker interpretation, and links to MediNova's 9 screening tools.
      2. UNIVERSAL VERSATILITY: Answers ANY user query politely, accurately, and thoroughly.
    """
    gemini_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    openai_key = os.environ.get('OPENAI_API_KEY')

    if not gemini_key and not openai_key:
        return None

    # Contextual active diagnosis snapshot
    diag_context_str = json.dumps(last_diagnosis if last_diagnosis else page_context, default=str)

    system_prompt = f"""You are MediNova Clinical AI Copilot & Medical Advisor, an advanced medical decision support and conversational AI integrated into the MediNova CDSS platform.

CORE OPERATIONAL MANDATES:
1. FIRST PRIORITY — CLINICAL & MEDICAL ACCURACY:
   - When the user asks medical, symptom, disease, drug, anatomical, laboratory, or healthcare questions:
     * Provide evidence-based, compassionate, structured medical information.
     * Emphasize clinical red flags and emergency guidance immediately if life-threatening symptoms are reported (call 112 / 911 / 108).
     * Actively cross-reference MediNova's 9 active diagnostic screening suites when relevant, providing markdown links:
       - Cardiovascular Disease: [Heart Screening](/heart) (13 vitals, BP, cholesterol, Oldpeak, ST depression)
       - Diabetes Mellitus: [Diabetes Screening](/diabetes) (HbA1c, glucose, BMI, insulin resistance)
       - Chronic Kidney Disease: [Kidney Screening](/kidney) (24 markers, creatinine, blood urea, albumin)
       - Hepatic Disorders: [Liver Screening](/liver) (10 enzymes, bilirubin, ALT, AST, ALP, A/G ratio)
       - Breast Oncology Cytology: [Breast Cancer Screening](/breast_cancer) (FNA nuclear morphometry)
       - Thyroid Dysfunction: [Thyroid Screening](/thyroid) (TSH, T3, TT4, T4U, FTI calibrated model)
       - Cardiac Arrhythmia: [Arrhythmia ECG Telemetry](/arrhythmia) (360 Hz Lead MLII, Pan-Tompkins QRS)
       - Malaria Parasitology: [Malaria Smear Scan](/malaria) (thin blood smear deep learning CNN)
       - Pneumonia Radiography: [Pneumonia CXR Scan](/pneumonia) (chest radiograph deep learning CNN)
       - Specialist Consultations: [Care Network & Appointments](/contact)
     * Explain laboratory units and standard physiological reference ranges clearly.
     * Maintain ethical disclaimers: MediNova is an AI clinical decision support tool for screening and education, not a replacement for in-person medical diagnosis.

2. UNIVERSAL CONVERSATIONAL CAPABILITY ("WHATEVER USER ASKS, REPLY"):
   - When the user asks non-medical, casual, general knowledge, science, programming, math, everyday, or personal questions:
     * Answer thoroughly, accurately, creatively, and politely. Never refuse or say "I only answer medical questions".
     * At the end of non-medical answers, add a short, warm footnote reminding the user that as MediNova's Clinical Copilot, you are always available if they need health or medical assistance.

CURRENT PATIENT SESSION CONTEXT:
- Active Web Page: {current_page}
- Active Diagnostic Context: {diag_context_str}
"""

    # 1. Try Google Gemini 3.8 Flash via google-genai SDK
    if gemini_key:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=gemini_key)

            # Build conversational history if available
            contents: List[Any] = []
            if chat_history and isinstance(chat_history, list):
                for turn in chat_history[-6:]:
                    role = "user" if turn.get("role") in ["user", "patient"] else "model"
                    txt = turn.get("content", turn.get("text", "")).strip()
                    if txt:
                        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=txt)]))

            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_msg)]))

            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
                max_output_tokens=1500
            )

            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=contents,
                config=config
            )

            if response and response.text:
                return {
                    "reply": response.text.strip(),
                    "confidence_badge": "MediNova Copilot • Gemini 3.8 Flash AI",
                    "engine": "gemini"
                }
        except Exception as e:
            logger.warning(f"google-genai SDK call failed, falling back to REST/local: {e}")

        # REST Fallback for Gemini 3.8 Flash
        try:
            import requests
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={gemini_key}"
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": f"{system_prompt}\n\nUser Question: {user_msg}"}]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 1500
                }
            }
            res = requests.post(url, json=payload, timeout=8)
            if res.status_code == 200:
                data = res.json()
                text = data['candidates'][0]['content']['parts'][0]['text']
                return {
                    "reply": text.strip(),
                    "confidence_badge": "MediNova Copilot • Gemini Live AI",
                    "engine": "gemini_rest"
                }
        except Exception as err:
            logger.warning(f"Gemini REST call failed: {err}")

    # 2. Try OpenAI GPT-4o-mini
    if openai_key:
        try:
            import requests
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
            messages = [{"role": "system", "content": system_prompt}]
            if chat_history and isinstance(chat_history, list):
                for turn in chat_history[-6:]:
                    r = "user" if turn.get("role") in ["user", "patient"] else "assistant"
                    t = turn.get("content", turn.get("text", "")).strip()
                    if t:
                        messages.append({"role": r, "content": t})
            messages.append({"role": "user", "content": user_msg})

            payload = {
                "model": "gpt-4o-mini",
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 1200
            }
            res = requests.post(url, json=payload, headers=headers, timeout=8)
            if res.status_code == 200:
                data = res.json()
                text = data['choices'][0]['message']['content']
                return {
                    "reply": text.strip(),
                    "confidence_badge": "MediNova Copilot • OpenAI Live AI",
                    "engine": "openai"
                }
        except Exception as err:
            logger.warning(f"OpenAI API call failed: {err}")

    return None

# ==============================================================================
# 8. CLINICAL & KNOWLEDGE BASE HELPER UTILITIES
# ==============================================================================

def detect_clinical_domain(message: str) -> Optional[str]:
    """Identifies matching clinical specialty domain from patient message."""
    msg = (message or '').lower()
    for domain_key in ['heart', 'diabetes', 'thyroid', 'arrhythmia', 'kidney', 'liver', 'breast_cancer', 'malaria', 'pneumonia', 'general']:
        domain_info = CLINICAL_DOMAINS[domain_key]
        if any(kw in msg for kw in domain_info['keywords']):
            return domain_key
    return None

def extract_form_auto_fill(message: str, current_page: str) -> Optional[Dict[str, float]]:
    """Extracts numerical vitals from user chat to auto-populate diagnostic forms."""
    msg = message.lower()
    page = (current_page or '').lower()

    if 'heart' in page:
        fill_data = {}
        m = re.search(r'\b(?:age|years?|yo)\s*[:=]?\s*(\d{1,3})\b', msg)
        if m: fill_data['age'] = float(m.group(1))
        m = re.search(r'\b(?:bp|blood\s*pressure|resting\s*bp)\s*[:=]?\s*(\d{2,3})\b', msg)
        if m: fill_data['resting_bp'] = float(m.group(1))
        m = re.search(r'\b(?:chol|cholesterol)\s*[:=]?\s*(\d{2,3})\b', msg)
        if m: fill_data['cholesterol'] = float(m.group(1))
        m = re.search(r'\b(?:hr|heart\s*rate|max\s*hr|pulse)\s*[:=]?\s*(\d{2,3})\b', msg)
        if m: fill_data['max_hr'] = float(m.group(1))
        m = re.search(r'\b(?:oldpeak|st\s*depression)\s*[:=]?\s*(\d+(?:\.\d+)?)\b', msg)
        if m: fill_data['oldpeak'] = float(m.group(1))
        return fill_data if len(fill_data) >= 2 else None

    if 'diabetes' in page:
        fill_data = {}
        m = re.search(r'\b(?:age|years?|yo)\s*[:=]?\s*(\d{1,3})\b', msg)
        if m: fill_data['age'] = float(m.group(1))
        m = re.search(r'\b(?:glucose|blood\s*sugar|fbs|fpg)\s*[:=]?\s*(\d{2,3})\b', msg)
        if m: fill_data['blood_glucose_level'] = float(m.group(1))
        m = re.search(r'\b(?:hba1c|a1c)\s*[:=]?\s*(\d+(?:\.\d+)?)\b', msg)
        if m: fill_data['HbA1c_level'] = float(m.group(1))
        m = re.search(r'\b(?:bmi)\s*[:=]?\s*(\d+(?:\.\d+)?)\b', msg)
        if m: fill_data['bmi'] = float(m.group(1))
        return fill_data if len(fill_data) >= 2 else None

    if 'thyroid' in page:
        fill_data = {}
        m = re.search(r'\b(?:age|years?|yo)\s*[:=]?\s*(\d{1,3})\b', msg)
        if m: fill_data['age'] = float(m.group(1))
        m = re.search(r'\b(?:tsh)\s*[:=]?\s*(\d+(?:\.\d+)?)\b', msg)
        if m: fill_data['TSH'] = float(m.group(1))
        m = re.search(r'\b(?:t3)\s*[:=]?\s*(\d+(?:\.\d+)?)\b', msg)
        if m: fill_data['T3'] = float(m.group(1))
        m = re.search(r'\b(?:tt4)\s*[:=]?\s*(\d+(?:\.\d+)?)\b', msg)
        if m: fill_data['TT4'] = float(m.group(1))
        m = re.search(r'\b(?:fti)\s*[:=]?\s*(\d+(?:\.\d+)?)\b', msg)
        if m: fill_data['FTI'] = float(m.group(1))
        return fill_data if len(fill_data) >= 2 else None

    if 'kidney' in page:
        fill_data = {}
        m = re.search(r'\b(?:bp|blood\s*pressure)\s*[:=]?\s*(\d{2,3})\b', msg)
        if m: fill_data['bp'] = float(m.group(1))
        m = re.search(r'\b(?:creatinine|sc)\s*[:=]?\s*(\d+(?:\.\d+)?)\b', msg)
        if m: fill_data['sc'] = float(m.group(1))
        m = re.search(r'\b(?:urea|bu)\s*[:=]?\s*(\d{1,3})\b', msg)
        if m: fill_data['bu'] = float(m.group(1))
        m = re.search(r'\b(?:hemo|hemoglobin)\s*[:=]?\s*(\d+(?:\.\d+)?)\b', msg)
        if m: fill_data['hemo'] = float(m.group(1))
        return fill_data if len(fill_data) >= 2 else None

    if 'liver' in page:
        fill_data = {}
        m = re.search(r'\b(?:bilirubin|total\s*bilirubin)\s*[:=]?\s*(\d+(?:\.\d+)?)\b', msg)
        if m: fill_data['Total_Bilirubin'] = float(m.group(1))
        m = re.search(r'\b(?:alt|sgpt)\s*[:=]?\s*(\d{1,4})\b', msg)
        if m: fill_data['Alamine_Aminotransferase'] = float(m.group(1))
        m = re.search(r'\b(?:ast|sgot)\s*[:=]?\s*(\d{1,4})\b', msg)
        if m: fill_data['Aspartate_Aminotransferase'] = float(m.group(1))
        m = re.search(r'\b(?:albumin)\s*[:=]?\s*(\d+(?:\.\d+)?)\b', msg)
        if m: fill_data['Albumin'] = float(m.group(1))
        return fill_data if len(fill_data) >= 2 else None

    return None

def parse_num(val_str: Any) -> Optional[float]:
    """Extracts first valid float from arbitrary string value."""
    if not isinstance(val_str, str):
        try:
            return float(val_str)
        except (ValueError, TypeError):
            return None
    m = re.search(r'[-+]?\d*\.?\d+', val_str)
    if m:
        try:
            return float(m.group(0))
        except (ValueError, TypeError):
            return None
    return None

def is_clinical_query(message: str) -> bool:
    """Detects whether a user message is medical or clinical in nature."""
    msg = (message or '').lower()
    
    # 1. Red flags
    if any(trig in msg for trig in EMERGENCY_TRIGGERS):
        return True
    
    # 2. Clinical domains
    if detect_clinical_domain(msg):
        return True
    
    # 3. Medications
    for med_data in MEDICATIONS_KB.values():
        if any(kw in msg for kw in med_data['keywords']):
            return True
            
    # 4. Extended symptoms
    for sym_data in EXTENDED_SYMPTOMS_KB.values():
        if any(kw in msg for kw in sym_data['keywords']):
            return True
            
    # 5. Biomarkers & explainers
    if any(term_key in msg for term_key in MEDICAL_EXPLAINERS):
        return True
        
    # 6. Clinical vocabulary terms
    clinical_keywords = [
        'health', 'doctor', 'hospital', 'medicine', 'disease', 'symptom', 'pain', 'ache',
        'blood', 'sugar', 'glucose', 'bp', 'pressure', 'cardiac', 'heart', 'liver', 'kidney',
        'urine', 'fever', 'cough', 'swelling', 'infection', 'cancer', 'tumor', 'biopsy',
        'ecg', 'ekg', 'scan', 'x-ray', 'radiology', 'ultrasound', 'hba1c', 'cholesterol',
        'prescription', 'dosage', 'tablet', 'pill', 'treatment', 'therapy', 'diet for',
        'nutrition', 'calories', 'weight loss', 'bmi', 'anemia', 'jaundice', 'vomiting',
        'diarrhea', 'constipation', 'fatigue', 'insomnia', 'headache', 'migraine', 'rash',
        'biomarker', 'tsh', 'creatinine', 'bilirubin', 'troponin', 'arrhythmia', 'malaria', 'pneumonia'
    ]
    if any(re.search(rf'\b{re.escape(kw)}\b', msg) for kw in clinical_keywords):
        return True

    return False

GENERAL_KNOWLEDGE_KB = {
    'photosynthesis': {
        'keywords': ['photosynthesis', 'how do plants make food', 'chloroplast', 'chlorophyll'],
        'title': 'Photosynthesis in Plants',
        'content': (
            "**Photosynthesis** is the fundamental biochemical process by which green plants, algae, and certain bacteria convert light energy into chemical energy stored in glucose.\n\n"
            "$$\\text{6CO}_2 + 6\\text{H}_2\\text{O} + \\text{Light Energy} \\xrightarrow{\\text{Chlorophyll}} \\text{C}_6\\text{H}_{12}\\text{O}_6 + 6\\text{O}_2$$\n\n"
            "- **Light Reactions:** Occur within the thylakoid membranes of chloroplasts, absorbing photons via chlorophyll to split water ($H_2O$), releasing oxygen ($O_2$), ATP, and NADPH.\n"
            "- **Calvin Cycle (Dark Reactions):** Occurs in the chloroplast stroma, fixing atmospheric carbon dioxide ($CO_2$) into carbohydrates using ATP and NADPH."
        )
    },
    'gravity': {
        'keywords': ['gravity', 'how does gravity work', 'gravitation', 'gravitational force'],
        'title': 'The Physics of Gravity',
        'content': (
            "**Gravity** is one of the four fundamental forces in physics:\n\n"
            "1. **Newtonian Gravity:** Every particle attracts every other particle with a force proportional to the product of their masses and inversely proportional to the square of the distance between them: $$F = G \\frac{m_1 m_2}{r^2}$$\n"
            "2. **Einstein's General Relativity:** Gravity is not a mechanical force but rather the curvature of four-dimensional spacetime caused by mass and energy. Massive bodies warp the fabric of spacetime, causing objects to follow geodesics (curved paths)."
        )
    },
    'shakespeare': {
        'keywords': ['romeo and juliet', 'shakespeare', 'who wrote hamlet', 'macbeth', 'william shakespeare'],
        'title': 'William Shakespeare & Classic Literature',
        'content': (
            "**William Shakespeare** (1564–1616) was an English playwright, poet, and actor widely regarded as the greatest writer in the English language.\n\n"
            "- **Romeo and Juliet:** Written early in Shakespeare's career (c. 1595), it is a tragedy about two young Italian star-crossed lovers whose deaths ultimately reconcile their feuding families (Montagues and Capulets).\n"
            "- **Major Tragedies:** *Hamlet*, *Othello*, *King Lear*, and *Macbeth*.\n"
            "- **Legacy:** Invented hundreds of words and phrases still standard in modern English."
        )
    },
    'artificial_intelligence': {
        'keywords': ['artificial intelligence', 'what is ai', 'machine learning', 'deep learning', 'neural network'],
        'title': 'Artificial Intelligence & Machine Learning',
        'content': (
            "**Artificial Intelligence (AI)** encompasses computational systems capable of performing tasks typically requiring human intelligence:\n\n"
            "- **Machine Learning (ML):** Algorithms (like XGBoost, Random Forests, Logistic Regression) that learn statistical patterns from data rather than explicit rule programming.\n"
            "- **Deep Learning (DL):** Artificial neural networks with multiple layers (CNNs, Transformers) modeled loosely after human visual cortex and cognitive processing, powering computer vision and large language models.\n"
            "- **MediNova AI Implementation:** MediNova utilizes deep Convolutional Neural Networks (CNNs) for radiographic imaging and tuned ensemble classifiers for clinical diagnostic screening."
        )
    }
}

# ==============================================================================
# 9. COMPREHENSIVE LOCAL CLINICAL & CONVERSATIONAL COPILOT ENGINE
# ==============================================================================

def medinova_clinical_copilot_engine(
    user_msg: str,
    current_page: str = '',
    page_context: Optional[Dict[str, Any]] = None,
    last_diagnosis: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Advanced local offline intelligence engine ensuring high-quality, clinical-first
    and versatile conversational responses without dead-ends.
    """
    msg = (user_msg or '').strip()
    msg_lower = msg.lower()
    page = (current_page or '').lower()
    page_context = page_context or {}

    # --------------------------------------------------------------------------
    # 1. EMERGENCY RED-FLAG PROTOCOL (HIGHEST PRIORITY)
    # --------------------------------------------------------------------------
    for trigger in EMERGENCY_TRIGGERS:
        if trigger in msg_lower:
            reply = (
                "### 🚨 HIGH-PRIORITY CLINICAL EMERGENCY PROTOCOL ALERT\n\n"
                f"**Immediate Safety Protocol Triggered:**\n"
                f"The acute presentation you reported (*\"{trigger}\"*) indicates a potential medical emergency requiring urgent attention.\n\n"
                "- **Call Emergency Services Immediately:** Dial **112 / 911 / 108** right now.\n"
                "- **Do Not Drive Yourself:** Request an urgent emergency paramedic ambulance.\n"
                "- **Sit in a Supported Position:** Remain seated upright with chest uncompressed.\n"
                "- **Loosen Restrictive Clothing:** Loosen collar, necktie, and belt to facilitate unlabored breathing.\n"
                "- **Calm Regulated Breathing:** Take slow, steady breaths while awaiting first responders.\n\n"
                "Visit our **[Emergency Directory & Hospital Finder](/contact)** for verified emergency departments nearby."
            )
            return {
                "success": True,
                "reply": reply,
                "confidence_badge": "Emergency Triage • Immediate Action",
                "is_emergency": True,
                "is_medical": True,
                "domain": "Emergency",
                "suggested_actions": [
                    {"label": "🚨 Emergency Directory & Care", "url": "/contact"},
                    {"label": "📞 Call 112 / 911", "url": "tel:911"}
                ],
                "follow_up_chips": [
                    "What first aid should I administer?",
                    "View nearest hospital emergency room",
                    "Clear chat session"
                ]
            }

    # --------------------------------------------------------------------------
    # 2. CONVERSATIONAL FORM AUTO-FILL
    # --------------------------------------------------------------------------
    fill_data = extract_form_auto_fill(msg, page)
    if fill_data:
        param_lines = "\n".join([f"- **`{k}`**: `{v}`" for k, v in fill_data.items()])
        clean_page_name = page.replace('/', '').replace('_', ' ').capitalize()
        reply = (
            f"### ⚡ MediNova Form Auto-Fill Activated\n\n"
            f"I parsed the patient clinical vitals from your message for **{clean_page_name} Screening**:\n\n"
            f"{param_lines}\n\n"
            "Click **Auto-Fill Form** below to inject these parameters directly into the diagnostic form with smooth highlight animations."
        )
        return {
            "success": True,
            "reply": reply,
            "confidence_badge": "Form Assistant • Auto-Fill Ready",
            "is_emergency": False,
            "is_medical": True,
            "domain": clean_page_name,
            "form_fill_data": fill_data,
            "suggested_actions": [
                {"label": "⚡ Auto-Fill Form with These Values", "action": "autofill"}
            ],
            "follow_up_chips": [
                "Run diagnostic analysis",
                "Load a normal reference case instead",
                "Explain the significance of these values"
            ]
        }

    # --------------------------------------------------------------------------
    # 3. GREETINGS & INTRODUCTIONS ("Whatever user ask reply")
    # --------------------------------------------------------------------------
    is_greeting = any(re.search(rf"\b{g}\b", msg_lower) for g in [
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
        'namaste', 'vanakkam', 'hola', 'sup', 'yo'
    ])
    if is_greeting and len(msg_lower.split()) <= 4:
        reply = (
            "### 🩺 Hello! Welcome to MediNova Clinical AI Copilot\n\n"
            "I am your dedicated medical decision support and healthcare advisor. I can help you with:\n\n"
            "1. **Symptom Triage & Guidance:** Tell me what symptoms you're feeling for immediate first-aid, safe diet tips, and physician questions.\n"
            "2. **9 AI Diagnostic Suites:** Screen for *Heart Disease, Diabetes, Kidney, Liver, Breast Oncology, Thyroid, Arrhythmia (ECG), Malaria, or Pneumonia*.\n"
            "3. **Lab Report Analysis:** Explain your lab test numbers (HbA1c, TSH, creatinine, liver enzymes, etc.) or scan report PDFs.\n"
            "4. **Care Network & Appointments:** Find accredited specialists near you and schedule priority consultations.\n\n"
            "How can I support your health or answer your questions today?"
        )
        return {
            "success": True,
            "reply": reply,
            "confidence_badge": "Clinical Advisor • Ready",
            "is_emergency": False,
            "is_medical": False,
            "domain": "General",
            "suggested_actions": [
                {"label": "🫀 Heart Screening", "url": "/heart"},
                {"label": "🩸 Diabetes Screening", "url": "/diabetes"},
                {"label": "🦋 Thyroid Screening", "url": "/thyroid"},
                {"label": "🔍 All 9 Suites", "url": "/features"}
            ],
            "follow_up_chips": [
                "I have high blood sugar or diabetes",
                "I have chest tightness or high BP",
                "Explain standard lab ranges",
                "How to book a specialist"
            ]
        }

    # --------------------------------------------------------------------------
    # 4. SYSTEM ARCHITECTURE, CREATOR & PLATFORM GUIDES
    # --------------------------------------------------------------------------
    if any(k in msg_lower for k in ['who made you', 'who created you', 'author', 'developer', 'credits', 'dharsan']):
        return {
            "success": True,
            "reply": GENERAL_CONVERSATION_RESPONSES['creator_info'],
            "confidence_badge": "System Architecture • Attribution",
            "is_emergency": False,
            "is_medical": False,
            "domain": "General",
            "suggested_actions": [
                {"label": "🔍 Explore 9 Suites", "url": "/features"},
                {"label": "🏥 View Care Network", "url": "/contact"}
            ],
            "follow_up_chips": [
                "What can MediNova do?",
                "How to book an appointment?",
                "Normal biomarker reference ranges"
            ]
        }

    if any(k in msg_lower for k in ['what can you do', 'features', 'help me with', 'overview', 'capabilities']):
        return {
            "success": True,
            "reply": GENERAL_CONVERSATION_RESPONSES['capabilities_overview'],
            "confidence_badge": "Platform Capabilities • Overview",
            "is_emergency": False,
            "is_medical": False,
            "domain": "General",
            "suggested_actions": [
                {"label": "🔍 Explore 9 Suites", "url": "/features"},
                {"label": "📄 Lab Report Scanner", "url": "/features#scanner"}
            ],
            "follow_up_chips": [
                "Who created MediNova?",
                "How to book an appointment?",
                "Normal biomarker reference ranges"
            ]
        }

    if any(k in msg_lower for k in ['book appointment', 'how to book', 'schedule consultation', 'find hospital', 'nearest clinic']):
        return {
            "success": True,
            "reply": GENERAL_CONVERSATION_RESPONSES['appointment_guide'],
            "confidence_badge": "Care Network • Consultation Booking",
            "is_emergency": False,
            "is_medical": True,
            "domain": "Appointments",
            "suggested_actions": [
                {"label": "🏥 Open Care Network & Booking", "url": "/contact"}
            ],
            "follow_up_chips": [
                "What specialists are available?",
                "Check symptoms before booking",
                "Explain my test results"
            ]
        }

    if any(k in msg_lower for k in ['scan report', 'how to scan', 'ocr', 'upload pdf', 'upload report']):
        return {
            "success": True,
            "reply": GENERAL_CONVERSATION_RESPONSES['scanner_guide'],
            "confidence_badge": "Report Scanner • Guidance",
            "is_emergency": False,
            "is_medical": True,
            "domain": "Scanner",
            "suggested_actions": [
                {"label": "📄 Open Lab Report Scanner", "url": "/features#scanner"}
            ],
            "follow_up_chips": [
                "What file formats are supported?",
                "What biomarkers can it extract?",
                "Check symptoms"
            ]
        }

    # --------------------------------------------------------------------------
    # 5. EXPLAIN PATIENT TEST RESULTS (BIOMARKER BREAKDOWN)
    # --------------------------------------------------------------------------
    active_diag = last_diagnosis or (page_context.get('last_diagnosis') if page_context else None)
    is_explain_query = any(k in msg_lower for k in [
        'explain my result', 'explain my test', 'explain my diagnosis', 'explain test result',
        'explain latest test', 'explain my latest test', 'explain results', 'my test result',
        'what caused my result', 'biomarker breakdown', 'biomarker analysis'
    ])
    if is_explain_query and active_diag:
        condition = active_diag.get('condition', 'Screening Analysis')
        prediction = active_diag.get('prediction', 'Assessment Complete')
        is_pos = active_diag.get('is_positive', False)
        summary = active_diag.get('patient_summary', [])
        specialist = active_diag.get('specialist', 'Clinical Specialist')
        urgency = active_diag.get('urgency_window', 'Within 24 to 48 hours')
        recommended_tests = active_diag.get('recommended_tests', [])

        biomarker_analysis = analyze_patient_biomarkers(condition, summary)
        analysis_lines = []
        for b in biomarker_analysis:
            status_icon = "🔴" if b['status'] == 'High Risk' else ("🟡" if b['status'] == 'Borderline' else "🟢")
            analysis_lines.append(f"- {status_icon} **{b['label']}** (`{b['value']}`): {b['note']} *(Target: {b['ref']})*")
        analysis_md = "\n".join(analysis_lines) if analysis_lines else "- Submitted parameters evaluated within clinical thresholds."

        reply = (
            f"### 🩺 MediNova AI Doctor: Diagnostic Result Explanation\n\n"
            f"**Screening Condition:** {condition}\n"
            f"**Algorithmic Assessment:** `{prediction}`\n"
            f"**Clinical Status:** **{'🚨 Elevated Clinical Risk Detected' if is_pos else '✅ Low / Normal Clinical Risk'}**\n\n"
            f"#### 📊 Biomarker Analysis & Targets:\n"
            f"{analysis_md}\n\n"
            f"#### 🏥 Recommended Specialist & Urgency:\n"
            f"- **Specialist:** **{specialist}**\n"
            f"- **Urgency Window:** **{urgency}**\n"
            f"- **Confirmatory Clinical Tests:** {', '.join(['`' + t + '`' for t in recommended_tests]) if recommended_tests else 'Routine primary care checkup'}\n\n"
            f"#### 📋 Next Clinical Steps:\n"
            f"{'Our validated machine learning model flagged elevated risk indicators. We recommend scheduling an in-person consultation with a specialist and bringing your recent test documentation.' if is_pos else 'Your submitted vitals and lab values fall within healthy physiological envelopes. Maintain routine preventive wellness screenings.'}"
        )
        return {
            "success": True,
            "reply": reply,
            "confidence_badge": f"Result Breakdown • {condition}",
            "is_emergency": is_pos,
            "is_medical": True,
            "domain": condition,
            "suggested_actions": [
                {"label": f"📄 View {condition} Suite", "url": active_diag.get('route', '/features')},
                {"label": "🏥 Book Specialist Appointment", "url": "/contact"}
            ],
            "follow_up_chips": [
                "What diet should I follow for this result?",
                "What questions should I ask my doctor?",
                "Normal biomarker reference ranges",
                "↺ Reset Consultation"
            ]
        }

    # --------------------------------------------------------------------------
    # 6. PHARMACOLOGY & MEDICATION SAFETY QUERIES
    # --------------------------------------------------------------------------
    for med_key, med_data in MEDICATIONS_KB.items():
        if any(kw in msg_lower for kw in med_data['keywords']):
            warnings_md = "\n".join([f"- ⚠️ {w}" for w in med_data['safety_warnings']])
            reply = (
                f"### 💊 Clinical Drug Information: {med_data['name']}\n\n"
                f"- **Pharmacological Class:** `{med_data['class']}`\n"
                f"- **Clinical Indication:** {med_data['indication']}\n"
                f"- **Standard Dosage Guidance:** {med_data['safe_dosage']}\n\n"
                f"#### ⚠️ Critical Safety & Contraindication Warnings:\n"
                f"{warnings_md}\n\n"
                f"> 💡 **Clinical Prescribing Note:** Medication regimens must be individualized by your treating physician based on your personal medical history, kidney/liver function, and potential drug-drug interactions."
            )
            return {
                "success": True,
                "reply": reply,
                "confidence_badge": f"Drug Reference • {med_data['name']}",
                "is_emergency": False,
                "is_medical": True,
                "domain": med_data['name'],
                "suggested_actions": [
                    {"label": f"🔬 Check Related {med_data['name']} Suite", "url": med_data['relevant_suite']},
                    {"label": "🏥 Consult a Physician", "url": "/contact"}
                ],
                "follow_up_chips": [
                    "What are common drug-drug interactions?",
                    "What foods affect this medication?",
                    "Check related lab biomarkers"
                ]
            }

    # --------------------------------------------------------------------------
    # 7. EXTENDED COMMON SYMPTOMS (HEADACHE, FEVER, STOMACH, BACK PAIN, ETC.)
    # --------------------------------------------------------------------------
    for sym_key, sym_data in EXTENDED_SYMPTOMS_KB.items():
        if any(kw in msg_lower for kw in sym_data['keywords']):
            red_flags_md = "\n".join([f"- 🚨 {rf}" for rf in sym_data['red_flags']])
            home_care_md = "\n".join([f"- {hc}" for hc in sym_data['home_care']])
            reply = (
                f"### 🩺 MediNova Clinical Triage: {sym_data['title']}\n\n"
                f"{sym_data['overview']}\n\n"
                f"#### 🚨 Red-Flag Warning Signs (Seek Urgent Care If Present):\n"
                f"{red_flags_md}\n\n"
                f"#### ✅ Evidence-Based Immediate Self-Care:\n"
                f"{home_care_md}\n\n"
                f"> 💡 **Clinical Reminder:** If symptoms persist, intensify, or interfere with daily function, consult a licensed healthcare professional for complete physical examination."
            )
            return {
                "success": True,
                "reply": reply,
                "confidence_badge": f"Symptom Triage • {sym_data['title']}",
                "is_emergency": False,
                "is_medical": True,
                "domain": sym_data['title'],
                "suggested_actions": [
                    {"label": sym_data['action_label'], "url": sym_data['action_route']},
                    {"label": "🔍 Explore All 9 Suites", "url": "/features"}
                ],
                "follow_up_chips": [
                    "What questions should I ask my doctor?",
                    "What over-the-counter medication is safe?",
                    "Check related organ functions"
                ]
            }

    # --------------------------------------------------------------------------
    # 8. BIOMARKER & CLINICAL EXPLAINERS
    # --------------------------------------------------------------------------
    for term_key, term_info in MEDICAL_EXPLAINERS.items():
        if term_key in msg_lower:
            reply = (
                f"### 📖 MediNova Clinical Explainer: {term_info['term']}\n\n"
                f"{term_info['explanation']}\n\n"
                f"> **Clinical Context:** In MediNova's diagnostic engine, this biomarker is cross-referenced against validated medical reference distributions to identify early metabolic and organ strain before irreversible disease occurs.\n\n"
                f"👉 **Screen Your Risk:** You can evaluate this biomarker directly in our **[{term_info.get('action_label', 'Diagnostic Suite')}]({term_info.get('route', '/features')})** module."
            )
            return {
                "success": True,
                "reply": reply,
                "confidence_badge": f"Clinical Knowledge • {term_info['term']}",
                "is_emergency": False,
                "is_medical": True,
                "domain": term_info.get('domain', 'Biomarker Explainer'),
                "suggested_actions": [
                    {"label": term_info.get('action_label', '🔍 Explore Diagnostic Suite'), "url": term_info.get('route', '/features')},
                    {"label": "🏥 Book Specialist Appointment", "url": "/contact"}
                ],
                "follow_up_chips": [
                    "What are the normal reference ranges?",
                    "Explain my test results",
                    "What foods help improve these numbers?"
                ]
            }

    # --------------------------------------------------------------------------
    # 9. CLINICAL DOMAIN TRIAGE & GUIDANCE (FIRST PRIORITY FOR MEDICAL QUESTIONS)
    # --------------------------------------------------------------------------
    matched_domain_key = detect_clinical_domain(msg)
    if matched_domain_key and matched_domain_key in CLINICAL_DOMAINS:
        d_info = CLINICAL_DOMAINS[matched_domain_key]
        care_guidance = d_info.get('immediate_care', '')
        q1_text = d_info.get('q1_text', '')
        
        reply = (
            f"### 🩺 MediNova Clinical Guidance: {d_info['name']}\n\n"
            f"Regarding your inquiry on **{d_info['name']}**, here is evidence-based clinical guidance:\n\n"
            f"{care_guidance}\n\n"
            f"#### 🏥 Matching MediNova AI Screening Module:\n"
            f"You can screen your diagnostic parameters right now using our **[{d_info['action_label']}]({d_info['route']})** module.\n\n"
            f"👉 **Next Step:** {q1_text}"
        )
        return {
            "success": True,
            "reply": reply,
            "confidence_badge": f"Clinical Guidance • {d_info['name']}",
            "is_emergency": False,
            "is_medical": True,
            "domain": d_info['name'].split('&')[0].strip(),
            "suggested_actions": [
                {"label": d_info['action_label'], "url": d_info['route']},
                {"label": "🏥 Book Specialist Appointment", "url": "/contact"}
            ],
            "follow_up_chips": d_info.get('q1_chips', [
                "What diet is best for this?",
                "What questions should I ask my doctor?",
                "Normal biomarker reference ranges"
            ])
        }

    # --------------------------------------------------------------------------
    # 10. STANDARD CLINICAL REFERENCE RANGES TABLE
    # --------------------------------------------------------------------------
    if any(k in msg_lower for k in ['reference range', 'normal range', 'normal values', 'what is normal', 'biomarker range']):
        reply = (
            "### 🧪 Standard Clinical Biomarker Reference Ranges\n\n"
            "MediNova evaluates submitted lab markers against standard clinical targets:\n\n"
            "| Domain | Biomarker | Optimal / Normal Target | Clinical Alert Cutoff |\n"
            "| :--- | :--- | :--- | :--- |\n"
            "| **Cardiovascular** | Resting Blood Pressure | `< 120/80 mm Hg` | `≥ 130/80 mm Hg` (Hypertension) |\n"
            "| **Cardiovascular** | Total Cholesterol | `< 200 mg/dL` | `≥ 240 mg/dL` (Hypercholesterolemia) |\n"
            "| **Cardiovascular** | ST Depression (Oldpeak) | `0.0 - 0.5 mm` | `> 1.5 mm` (Subendocardial Ischemia) |\n"
            "| **Glycemic** | Fasting Blood Glucose | `70 - 99 mg/dL` | `≥ 126 mg/dL` (Diabetes Mellitus) |\n"
            "| **Glycemic** | Glycated Hemoglobin (HbA1c) | `< 5.7%` | `≥ 6.5%` (Diabetes Mellitus) |\n"
            "| **Thyroid** | TSH | `0.4 - 4.0 mIU/L` | `> 4.5 mIU/L` (Hypothyroidism) |\n"
            "| **Thyroid** | Free T4 | `0.8 - 1.8 ng/dL` | `< 0.8 ng/dL` (Thyroid Deficiency) |\n"
            "| **Renal** | Serum Creatinine | `0.7 - 1.2 mg/dL` (M) | `> 1.5 mg/dL` (Impaired GFR) |\n"
            "| **Renal** | Blood Urea | `15 - 45 mg/dL` | `> 50 mg/dL` (Azotemia) |\n"
            "| **Hepatic** | Total Bilirubin | `0.2 - 1.2 mg/dL` | `> 2.0 mg/dL` (Hyperbilirubinemia) |\n"
            "| **Hepatic** | ALT (SGPT) | `< 40 U/L` | `> 50 U/L` (Hepatocellular Strain) |\n"
            "| **Hepatic** | AST (SGOT) | `< 40 U/L` | `> 50 U/L` (Hepatic/Cardiac Strain) |\n\n"
            "> 💡 **Clinical Note:** Laboratory findings must always be correlated with physical signs and clinical history by a licensed physician."
        )
        return {
            "success": True,
            "reply": reply,
            "confidence_badge": "Clinical Standards • Reference Ranges",
            "is_emergency": False,
            "is_medical": True,
            "domain": "Reference Ranges",
            "suggested_actions": [
                {"label": "🫀 Heart Screening", "url": "/heart"},
                {"label": "🩸 Diabetes Screening", "url": "/diabetes"},
                {"label": "🦋 Thyroid Screening", "url": "/thyroid"},
                {"label": "🩺 Kidney Screening", "url": "/kidney"}
            ],
            "follow_up_chips": [
                "Which biomarkers are most dangerous when elevated?",
                "Check symptoms and recommend a test",
                "Explain my latest test results"
            ]
        }

    # --------------------------------------------------------------------------
    # 11. GENERAL KNOWLEDGE KB (SCIENCE, HISTORY, AI, TECH)
    # --------------------------------------------------------------------------
    for gk_key, gk_data in GENERAL_KNOWLEDGE_KB.items():
        if any(kw in msg_lower for kw in gk_data['keywords']):
            reply = (
                f"### 💡 {gk_data['title']}\n\n"
                f"{gk_data['content']}\n\n"
                "---\n"
                "> 🩺 **MediNova Clinical Copilot:** While I'm glad to answer your general and scientific questions, "
                "my primary clinical role is supporting your health! You can also ask me about symptoms, lab reports, "
                "or explore our 9 diagnostic screening suites ([Explore Features](/features))."
            )
            return {
                "success": True,
                "reply": reply,
                "confidence_badge": f"General Knowledge • {gk_data['title']}",
                "is_emergency": False,
                "is_medical": False,
                "domain": "General Knowledge",
                "suggested_actions": [
                    {"label": "🔍 Explore All 9 Suites", "url": "/features"},
                    {"label": "🏥 Care Network & Clinics", "url": "/contact"}
                ],
                "follow_up_chips": [
                    "Check my health symptoms",
                    "Explain standard biomarker ranges",
                    "How do I scan a lab report?"
                ]
            }

    # --------------------------------------------------------------------------
    # 12. UNIVERSAL NON-MEDICAL FALLBACK ("Whatever User Asks Reply")
    # --------------------------------------------------------------------------
    is_med = is_clinical_query(msg)
    clean_q = msg.replace('?', '').replace('!', '').strip()
    reply = (
        f"### 🤖 MediNova Clinical AI Advisor\n\n"
        f"Thank you for your question: *\"{msg}\"*\n\n"
        "Here is what you should know:\n\n"
        f"- **Direct Answer:** Regarding *{clean_q}*, I am designed to assist you with comprehensive information. "
        "Whether you are inquiring about everyday wellness, healthy lifestyle habits, biology, technology, or scientific concepts, "
        "maintaining healthy, balanced habits and evidence-backed knowledge is always paramount.\n\n"
        "- **First Priority — Healthcare & Medical Support:**\n"
        "As your MediNova clinical assistant, my primary specialization is helping you understand your health, evaluate symptoms, "
        "interpret medical lab reports, and access our 9 validated clinical screening models:\n"
        "  - **Cardiovascular & Blood Pressure:** [Cardiovascular Screening](/heart)\n"
        "  - **Metabolic & Glycemic Health:** [Diabetes Screening](/diabetes)\n"
        "  - **Thyroid & Endocrine Function:** [Thyroid Screening](/thyroid)\n"
        "  - **Cardiac Arrhythmia & ECG:** [Arrhythmia Telemetry](/arrhythmia)\n"
        "  - **Renal, Hepatic & Oncological Panels:** [All Diagnostic Suites](/features)\n"
        "  - **Verified Specialists & Appointments:** [Care Network & Hospital Booking](/contact)\n\n"
        "> 🩺 *If you have any specific health symptoms, medications, or lab values you would like me to review, please ask!*"
    )
    return {
        "success": True,
        "reply": reply,
        "confidence_badge": "Conversational Advisor • General Assistance",
        "is_emergency": False,
        "is_medical": is_med,
        "domain": "Medical Inquiry" if is_med else "General Conversation",
        "suggested_actions": [
            {"label": "🫀 Heart Screening", "url": "/heart"},
            {"label": "🩸 Diabetes Screening", "url": "/diabetes"},
            {"label": "🦋 Thyroid Screening", "url": "/thyroid"},
            {"label": "🏥 Find Specialist Hospitals", "url": "/contact"}
        ],
        "follow_up_chips": [
            "Check my symptoms",
            "Explain standard biomarker ranges",
            "How do I scan a lab report?",
            "What can MediNova do?"
        ]
    }

def analyze_patient_biomarkers(condition: str, patient_summary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evaluates submitted patient biomarkers against medical reference standards across all 9 modules."""
    breakdown = []
    condition_lower = condition.lower() if condition else ''

    for item in patient_summary:
        label = item.get('label', '')
        val_str = str(item.get('value', ''))
        num_val = parse_num(val_str)
        status = 'Normal'
        ref = 'Standard Clinical Target'
        note = 'Within expected physiological limits.'

        if 'heart' in condition_lower:
            if 'Blood Pressure' in label:
                ref = '< 120 mm Hg'
                if num_val and num_val >= 140:
                    status = 'High Risk'
                    note = f'{int(num_val)} mm Hg indicates Stage 2 Hypertension.'
                elif num_val and num_val >= 130:
                    status = 'Borderline'
                    note = f'{int(num_val)} mm Hg indicates Stage 1 Hypertension.'
            elif 'Cholesterol' in label:
                ref = '< 200 mg/dL'
                if num_val and num_val >= 240:
                    status = 'High Risk'
                    note = f'{int(num_val)} mg/dL indicates hypercholesterolemia.'
                elif num_val and num_val >= 200:
                    status = 'Borderline'
                    note = f'{int(num_val)} mg/dL is borderline elevated.'
            elif 'Oldpeak' in label or 'ST Depression' in label:
                ref = '< 1.0 mm'
                if num_val and num_val >= 1.5:
                    status = 'High Risk'
                    note = f'{num_val} mm suggests exercise-induced subendocardial ischemia.'
            elif 'Thallium' in label:
                ref = 'Normal Perfusion'
                if 'defect' in val_str.lower():
                    status = 'High Risk'
                    note = 'Myocardial perfusion defect detected.'

        elif 'diabetes' in condition_lower:
            if 'HbA1c' in label:
                ref = '< 5.7%'
                if num_val and num_val >= 6.5:
                    status = 'High Risk'
                    note = f'{num_val}% indicates diabetic glycemic threshold over the past 90 days.'
                elif num_val and num_val >= 5.7:
                    status = 'Borderline'
                    note = f'{num_val}% falls in the impaired prediabetic spectrum.'
            elif 'Blood Glucose' in label:
                ref = '70 - 99 mg/dL'
                if num_val and num_val >= 126:
                    status = 'High Risk'
                    note = f'{int(num_val)} mg/dL exceeds the clinical diagnostic fasting cutoff.'
                elif num_val and num_val >= 100:
                    status = 'Borderline'
                    note = f'{int(num_val)} mg/dL indicates impaired fasting glucose.'
            elif 'BMI' in label:
                ref = '18.5 - 24.9 kg/m²'
                if num_val and num_val >= 30.0:
                    status = 'High Risk'
                    note = f'{num_val} kg/m² indicates obesity-associated insulin resistance.'

        elif 'thyroid' in condition_lower:
            if 'TSH' in label:
                ref = '0.4 - 4.0 mIU/L'
                if num_val and num_val >= 4.5:
                    status = 'High Risk'
                    note = f'{num_val} mIU/L indicates elevated TSH (Hypothyroidism).'
                elif num_val and num_val < 0.4:
                    status = 'High Risk'
                    note = f'{num_val} mIU/L indicates suppressed TSH (Hyperthyroidism).'
            elif 'TT4' in label or 'T4' in label:
                ref = '4.5 - 12.0 µg/dL'
                if num_val and num_val < 4.5:
                    status = 'High Risk'
                    note = f'{num_val} µg/dL reflects low circulating thyroxine.'

        elif 'arrhythmia' in condition_lower:
            if 'Heart Rate' in label or 'BPM' in label:
                ref = '60 - 100 bpm'
                if num_val and num_val > 100:
                    status = 'High Risk'
                    note = f'{int(num_val)} bpm indicates tachycardia.'
                elif num_val and num_val < 60:
                    status = 'Borderline'
                    note = f'{int(num_val)} bpm indicates bradycardia.'
            elif 'PVC' in label or 'Ectopy' in label:
                ref = '0 Ectopic Beats'
                if 'detected' in val_str.lower() or (num_val and num_val > 0):
                    status = 'High Risk'
                    note = 'Premature ventricular ectopic beats identified in rhythm strip.'

        elif 'kidney' in condition_lower:
            if 'Creatinine' in label:
                ref = '0.7 - 1.2 mg/dL'
                if num_val and num_val > 1.2:
                    status = 'High Risk'
                    note = f'{num_val} mg/dL reflects decreased glomerular filtration clearance.'
            elif 'Blood Urea' in label:
                ref = '15 - 45 mg/dL'
                if num_val and num_val > 45:
                    status = 'High Risk'
                    note = f'{num_val} mg/dL indicates retention of nitrogenous metabolic waste.'

        elif 'liver' in condition_lower:
            if 'Total Bilirubin' in label:
                ref = '0.1 - 1.2 mg/dL'
                if num_val and num_val > 1.2:
                    status = 'High Risk'
                    note = f'{num_val} mg/dL indicates hyperbilirubinemia.'
            elif 'ALT' in label:
                ref = '7 - 56 U/L'
                if num_val and num_val > 56:
                    status = 'High Risk'
                    note = f'{int(num_val)} U/L indicates acute or chronic hepatocellular injury.'

        elif 'breast' in condition_lower:
            if 'Radius' in label:
                ref = '< 14.0 µm'
                if num_val and num_val >= 14.5:
                    status = 'High Risk'
                    note = f'{num_val} µm reflects nuclear enlargement.'

        elif 'malaria' in condition_lower:
            if 'Classification' in label:
                ref = 'Uninfected (Negative)'
                if 'parasitized' in val_str.lower() or 'infected' in val_str.lower():
                    status = 'High Risk'
                    note = 'Intracellular Plasmodium trophozoites detected in red blood cells.'

        elif 'pneumonia' in condition_lower:
            if 'Assessment' in label:
                ref = 'Clear / Unremarkable'
                if 'consolidation' in val_str.lower() or 'infiltrates' in val_str.lower():
                    status = 'High Risk'
                    note = 'Parenchymal consolidation and airspace opacification identified.'

        breakdown.append({
            'label': label,
            'value': val_str,
            'ref': ref,
            'status': status,
            'note': note
        })

    return breakdown


# ==============================================================================
# 10. MULTI-TURN ACTION PLAN & RESULT EXPLANATION GENERATORS
# ==============================================================================

def generate_situation_action_plan(
    domain_key: str,
    severity: Optional[str],
    associated_symptoms: Optional[str],
    disease_insights: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generates a structured, evidence-based clinical action plan for the patient."""
    d_info = CLINICAL_DOMAINS.get(domain_key, CLINICAL_DOMAINS['general'])
    insight_key = d_info.get('insight_key', 'general')

    sev_str = (severity or '').lower()
    assoc_str = (associated_symptoms or '').lower()
    is_high_risk = any(w in sev_str for w in ['severe', 'sudden', 'high', 'critical', 'red', '🔴', '103', '250', 'bad']) or \
                   any(w in assoc_str for w in ['yes', 'shortness of breath', 'severe', 'pain', 'sweating', 'arm', 'jaw', 'low oxygen', 'chills'])

    immediate_section = d_info['emergency_advice'] if is_high_risk else d_info['immediate_care']

    diet_section = ""
    insights = disease_insights or {}
    if insight_key in insights:
        disease_dict = insights[insight_key]
        sub_key = 'malignant' if 'malignant' in disease_dict else ('infected' if 'infected' in disease_dict else 'positive')
        insight_data = disease_dict.get(sub_key, {})

        dos = insight_data.get('diet_dos', [])
        donts = insight_data.get('diet_donts', [])
        recs = insight_data.get('recommendations', [])
        doc_qs = insight_data.get('doctor_questions', [])

        diet_section = (
            "\n\n#### 🥗 3. Safe Diet & Care Protocol\n"
            "**✅ Recommended Evidence-Based Nutritional Dos:**\n"
            + "\n".join([f"- {item}" for item in dos[:3]]) + "\n\n"
            "**❌ Foods & Habits to Strictly Avoid:**\n"
            + "\n".join([f"- {item}" for item in donts[:3]]) + "\n\n"
            "**📋 Clinical Recommendations:**\n"
            + "\n".join([f"- {item}" for item in recs[:3]]) + "\n\n"
            "**❓ High-Yield Questions for Your Physician:**\n"
            + "\n".join([f"- \"{item}\"" for item in doc_qs[:3]])
        )

    reply = (
        f"### 🩺 MediNova Situation-Handling Action Plan: {d_info['name']}\n\n"
        f"Based on your clinical consultation (*Severity: {severity or 'Noted'}* | *Associated Indicators: {associated_symptoms or 'Evaluated'}*), here is your tailored step-by-step action plan:\n\n"
        f"#### 🚨 1. Immediate Action — How to Handle This Right Now\n"
        f"{immediate_section}\n\n"
        f"#### 🩺 2. Recommended MediNova AI Screening\n"
        f"- **Recommended Modality:** **{d_info['name']}**\n"
        f"- **Direct Route:** Click the screening button below to launch the automated AI model and test your clinical biomarkers.\n"
        f"{diet_section}\n\n"
        f"> 💡 **Clinical Reminder:** MediNova is an AI clinical decision support tool. If your symptoms worsen or you feel acute distress, contact emergency responders (**112 / 911 / 108**) immediately."
    )

    actions = [
        {"label": d_info['action_label'], "url": d_info['route']},
        {"label": "🏥 Emergency Directory & Specialists", "url": "/contact"}
    ]

    quick_replies = [
        "📊 Explain my latest test result",
        "🫀 Check cardiovascular symptoms",
        "🩸 Check blood sugar / diabetes",
        "🦋 Check thyroid / fatigue",
        "↺ Reset Consultation"
    ]

    return {
        "reply": reply,
        "quick_replies": quick_replies,
        "actions": actions,
        "confidence_badge": f"Action Plan • {d_info['name']}",
        "is_emergency": is_high_risk,
        "suggested_actions": actions,
        "follow_up_chips": quick_replies
    }


def handle_explain_test_result(last_diagnosis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Generates an in-depth breakdown of the most recent diagnostic result from the user's session."""
    if not last_diagnosis:
        reply = (
            "### 📊 MediNova Diagnostic Results Analysis\n\n"
            "No completed diagnostic screening was detected in your active session.\n\n"
            "**How to run a screening and receive an instant AI biomarker analysis:**\n"
            "1. Choose any of MediNova's **9 clinical diagnostic suites** below.\n"
            "2. Enter patient biomarkers, or click **⚡ Fill High-Risk / Healthy Sample** for instant test automation.\n"
            "3. Click **Submit** to run the validated machine learning model.\n"
            "4. On your result card, click **🤖 Ask MediNova AI to Explain This Result** for an automatic breakdown!\n\n"
            "Which diagnostic suite would you like to explore?"
        )
        actions = [
            {"label": "🫀 Heart Screening", "url": "/heart"},
            {"label": "🩸 Diabetes Screening", "url": "/diabetes"},
            {"label": "🦋 Thyroid Screening", "url": "/thyroid"},
            {"label": "⚡ Arrhythmia ECG", "url": "/arrhythmia"},
            {"label": "🔍 All 9 Suites", "url": "/features"}
        ]
        chips = [
            "🫀 Chest pain or high BP",
            "🩸 High blood sugar advice",
            "🦋 Thyroid or fatigue check",
            "🧪 Kidney / Liver check"
        ]
        return {
            "reply": reply,
            "quick_replies": chips,
            "actions": actions,
            "confidence_badge": "Session Context • No Active Test",
            "is_emergency": False,
            "suggested_actions": actions,
            "follow_up_chips": chips
        }

    condition = last_diagnosis.get('condition', 'Diagnostic Screening')
    route = last_diagnosis.get('route', '/features')
    prediction = last_diagnosis.get('prediction', 'Evaluated')
    is_pos = bool(last_diagnosis.get('is_positive', False))
    patient_summary = last_diagnosis.get('patient_summary', [])
    specialist = last_diagnosis.get('specialist', 'Clinical Specialist')
    urgency = last_diagnosis.get('urgency_window', 'Within 24-48 hours')
    recommended_tests = last_diagnosis.get('recommended_tests', [])

    biomarker_eval = analyze_patient_biomarkers(condition, patient_summary)
    summary_lines = []
    for item in biomarker_eval:
        icon = "🔴" if item['status'] == 'High Risk' else ("🟡" if item['status'] == 'Borderline' else "🟢")
        summary_lines.append(f"- {icon} **{item.get('label')}**: `{item.get('value')}` ({item.get('note')})")
    summary_md = "\n".join(summary_lines) if summary_lines else "- General scan parameters evaluated within clinical thresholds."

    status_badge = "🚨 High Clinical Alert / Elevated Risk" if is_pos else "✅ Optimal / Low Risk Detected"

    reply = (
        f"### 🩺 MediNova AI Doctor: Diagnostic Result Explanation\n\n"
        f"**Screening Condition:** {condition}\n"
        f"**Algorithmic Assessment:** `{prediction}`\n"
        f"**Clinical Status:** **{status_badge}**\n\n"
        f"#### 📊 Biomarker Breakdown Recorded for Patient:\n"
        f"{summary_md}\n\n"
        f"#### 🏥 Recommended Specialist & Urgency:\n"
        f"- **Specialist:** **{specialist}**\n"
        f"- **Urgency Window:** **{urgency}**\n"
        f"- **Confirmatory Clinical Tests:** {', '.join(['`' + t + '`' for t in recommended_tests]) if recommended_tests else 'Routine primary care checkup'}\n\n"
        f"#### 📋 How to Handle This Result:\n"
        f"{'Our machine learning models detected elevated physiological risk markers. Schedule an in-person clinical consultation with your specialist and assemble previous medical records for review.' if is_pos else 'All submitted biomarkers and imaging features align with healthy physiological reference ranges. Continue routine preventive wellness checkups.'}"
    )

    actions = [
        {"label": f"📄 Re-open {condition} Suite", "url": route},
        {"label": "🏥 Specialist & Emergency Directory", "url": "/contact"}
    ]
    quick_replies = [
        "What diet should I follow?",
        "What questions should I ask my doctor?",
        "Check another symptom",
        "↺ Reset Consultation"
    ]

    return {
        "reply": reply,
        "quick_replies": quick_replies,
        "actions": actions,
        "confidence_badge": f"Result Breakdown • {condition}",
        "is_emergency": is_pos,
        "suggested_actions": actions,
        "follow_up_chips": quick_replies
    }
