import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

app = Flask(__name__)

# Configure Upload & Sample Folders
UPLOAD_FOLDER = os.path.join('static', 'uploads')
SAMPLES_FOLDER = os.path.join('static', 'samples')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAMPLES_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SAMPLES_FOLDER'] = SAMPLES_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

# Load Pre-trained Models
print("Loading medical diagnostic models...")
MODELS_DIR = 'models'

def load_pkl_model(filename):
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None

heart_model = load_pkl_model('heart_disease_model.pkl')
diabetes_model = load_pkl_model('diabetes_model.pkl')
kidney_model = load_pkl_model('kidney_model.pkl')
liver_model = load_pkl_model('liver_model.pkl')
breast_cancer_model = load_pkl_model('breast_cancer_model.pkl')

pneumonia_model_path = os.path.join(MODELS_DIR, 'pneumonia_model_new.h5')
malaria_model_path = os.path.join(MODELS_DIR, 'malaria_cnn_model_fast.h5')

pneumonia_model = None
malaria_model = None

if os.path.exists(pneumonia_model_path):
    try:
        pneumonia_model = load_model(pneumonia_model_path)
        print("Pneumonia model loaded successfully.")
    except Exception as e:
        print(f"Error loading Pneumonia model: {e}")

if os.path.exists(malaria_model_path):
    try:
        malaria_model = load_model(malaria_model_path)
        print("Malaria model loaded successfully.")
    except Exception as e:
        print(f"Error loading Malaria model: {e}")

# ============================================================================
# COMPREHENSIVE CLINICAL INSIGHTS & EMERGENCY PROTOCOLS
# ============================================================================
DISEASE_INSIGHTS = {
    'heart_disease': {
        'positive': {
            'is_positive': True,
            'title': 'Cardiovascular Risk Indicators Detected',
            'severity': 'High Clinical Risk - Immediate Consultation Advised',
            'badge': 'ATTENTION REQUIRED',
            'status_color': 'danger',
            'description': 'AI algorithmic analysis identified significant cardiovascular markers (such as elevated ST depression, atypical chest pressure parameters, or abnormal resting ECG markers) that correlate with coronary artery disease risk.',
            'specialist': 'Board-Certified Cardiologist & Coronary Care Specialist',
            'urgency_window': 'Schedule clinical consultation within 24 to 48 hours.',
            'risk_factors': [
                'Elevated systolic or diastolic resting blood pressure',
                'Hypercholesterolemia (elevated serum cholesterol > 200 mg/dL)',
                'Exercise-induced angina or ischemic ST depression patterns',
                'Fluoroscopy showing multiple major vessels with calcification',
                'Underlying metabolic load or family history of myocardial infarction'
            ],
            'emergency_symptoms': [
                'Crushing, squeezing chest pressure or tightness radiating to left arm, neck, or jaw',
                'Sudden severe shortness of breath, cold sweats, or unprovoked nausea/dizziness',
                'Sudden irregular heart palpitations accompanied by lightheadedness or syncope (fainting)'
            ],
            'recommendations': [
                'Immediately schedule a comprehensive consultation with a cardiologist',
                'Undergo a 12-lead Electrocardiogram (ECG) and an Echocardiogram (Echo Doppler)',
                'Conduct cardiac enzyme biomarkers panel (Troponin I/T, CK-MB, hs-CRP)',
                'Refrain from high-intensity physical exertion or heavy lifting until cleared by a doctor',
                'Maintain an accurate twice-daily resting Blood Pressure and Heart Rate log'
            ],
            'doctor_questions': [
                'Do these clinical biomarkers indicate coronary artery stenosis or microvascular ischemia?',
                'Should I undergo a Coronary CT Angiography or nuclear myocardial perfusion scan?',
                'Are preventive cardioprotective medications (e.g., statins, ACE inhibitors, or beta blockers) indicated for me?',
                'What target resting systolic/diastolic blood pressure and LDL levels should I strictly maintain?'
            ],
            'diet_dos': [
                'Heart-healthy Mediterranean diet: high in leafy greens, avocados, walnuts, and extra virgin olive oil',
                'Omega-3 rich foods (salmon, flaxseeds, chia seeds) to lower systemic arterial inflammation',
                'High-soluble fiber foods (steel-cut oats, legumes, berries) to actively bind and lower serum cholesterol'
            ],
            'diet_donts': [
                'Strictly avoid high sodium (canned soups, processed meats, chips) - keep sodium < 1,500 mg/day',
                'Eliminate trans fats, hydrogenated oils, deep-fried snacks, and fatty cuts of red meat',
                'Avoid excessive caffeine, energy drinks, and tobacco products which cause acute arterial vasoconstriction'
            ],
            'recommended_tests': ['12-Lead ECG', 'Echocardiogram', 'Lipid Panel', 'Troponin-I', 'Coronary CT Angiogram']
        },
        'negative': {
            'is_positive': False,
            'title': 'Cardiovascular Profile Within Normal Limits',
            'severity': 'Low Risk Detected - Favorable Parameters',
            'badge': 'OPTIMAL HEALTH',
            'status_color': 'safe',
            'description': 'AI assessment detected no statistically significant risk factors for coronary artery disease based on the submitted clinical parameters.',
            'specialist': 'Primary Care Physician / Preventive Cardiologist',
            'urgency_window': 'Routine annual wellness screening recommended.',
            'tips': [
                'Maintain 150 minutes of moderate aerobic cardiovascular activity weekly (e.g., brisk walking, swimming)',
                'Continue maintaining balanced blood pressure below 120/80 mm Hg and cholesterol < 200 mg/dL',
                'Keep fasting blood glucose stable and maintain body mass index (BMI) within 18.5 - 24.9',
                'Perform an annual preventive wellness ECG and lipid panel screening'
            ],
            'diet_dos': [
                'Eat a vibrant rainbow of seasonal fruits, antioxidant-rich berries, and dark leafy vegetables',
                'Incorporate lean proteins (fish, poultry, legumes) and whole unrefined grains'
            ],
            'diet_donts': [
                'Avoid high-sodium fast food and sugar-sweetened beverages to preserve vascular elasticity'
            ],
            'recommended_tests': ['Annual Routine Wellness Lipid Panel', 'Routine Annual Resting ECG Checkup']
        }
    },
    'diabetes': {
        'positive': {
            'is_positive': True,
            'title': 'Elevated Glycemic & Metabolic Risk Detected',
            'severity': 'High Risk of Type 2 Diabetes / Dysglycemia',
            'badge': 'ATTENTION REQUIRED',
            'status_color': 'danger',
            'description': 'Based on submitted biomarkers (glycated hemoglobin HbA1c, fasting plasma glucose, BMI, and hypertension history), the AI model flagged positive indicators consistent with impaired glucose tolerance or clinical diabetes.',
            'specialist': 'Consult an Endocrinologist & Certified Diabetes Educator',
            'urgency_window': 'Schedule a medical consultation within 3 to 5 days.',
            'risk_factors': [
                'Elevated HbA1c level (indicating chronic hyperglycemia over the past 90 days)',
                'Elevated blood glucose levels exceeding normal fasting or post-prandial ranges',
                'Increased Body Mass Index (BMI > 25 kg/m² indicating excess visceral adiposity)',
                'Hypertension or co-existing cardiovascular strain amplifying insulin resistance'
            ],
            'emergency_symptoms': [
                'Sudden extreme thirst (polydipsia) paired with frequent urination (polyuria)',
                'Fruity breath odor, rapid deep breathing, confusion, or severe nausea (symptoms of Diabetic Ketoacidosis - DKA)',
                'Sudden blurred vision, unexplained dizziness, extreme fatigue, or numbness/tingling in hands/feet'
            ],
            'recommendations': [
                'Schedule a confirmatory venous blood draw for Fasting Plasma Glucose and Oral Glucose Tolerance Test (OGTT)',
                'Obtain a home glucometer to record fasting, 2-hour post-meal, and bedtime glucose values',
                'Consult a licensed clinical dietician for a personalized Low-Glycemic Index (GI) meal plan',
                'Engage in 20-30 minutes of daily post-meal moderate walking to boost peripheral insulin sensitivity',
                'Schedule baseline microvascular screenings (diabetic retinal exam and urine microalbuminuria test)'
            ],
            'doctor_questions': [
                'What is my definitive diagnosis: prediabetes, type 2 diabetes, or secondary dysglycemia?',
                'Do you recommend starting oral insulin-sensitizing therapy (such as Metformin) immediately?',
                'What is my individual target HbA1c percentage over the next 3 to 6 months?',
                'How often should I monitor my blood glucose throughout the day?'
            ],
            'diet_dos': [
                'Complex carbohydrates with high fiber: quinoa, brown rice, steel-cut oats, and legumes',
                'Non-starchy cruciferous vegetables: broccoli, spinach, kale, bell peppers, and cauliflower',
                'Healthy fats and lean proteins that stabilize postprandial glucose spikes'
            ],
            'diet_donts': [
                'Eliminate refined sugars, fruit juices, sodas, energy drinks, syrups, and sweetened bakery products',
                'Avoid refined carbohydrates: white bread, instant noodles, white rice, and sugary cereals',
                'Limit excessive tropical fruits with high fructose content (mangoes, ripe bananas, dried dates)'
            ],
            'recommended_tests': ['Venous Fasting Plasma Glucose', 'Oral Glucose Tolerance Test (OGTT)', 'C-Peptide Level', 'Lipid Panel', 'Urine Microalbumin/Creatinine Ratio']
        },
        'negative': {
            'is_positive': False,
            'title': 'Glycemic Markers Within Normal Range',
            'severity': 'Low Risk Detected - Optimal Glucose Regulation',
            'badge': 'OPTIMAL HEALTH',
            'status_color': 'safe',
            'description': 'Your clinical markers (HbA1c, glucose, and body mass metrics) show favorable insulin sensitivity with no indicators of active diabetes.',
            'specialist': 'General Physician / Preventive Health Specialist',
            'urgency_window': 'Annual routine metabolic panel recommended.',
            'tips': [
                'Maintain an active daily lifestyle with regular resistance and cardiovascular exercise',
                'Keep HbA1c below 5.7% and fasting blood sugar within 70 - 99 mg/dL',
                'Practice portion control and limit nighttime snacking on refined sweets',
                'Stay well-hydrated with pure water instead of sweetened sodas and juices'
            ],
            'diet_dos': [
                'Focus on whole foods, diverse dietary fibers, seeds, nuts, and clean lean proteins'
            ],
            'diet_donts': [
                'Avoid regular consumption of sugary beverages and processed ultra-refined snacks'
            ],
            'recommended_tests': ['Annual Fasting Blood Sugar Test', 'Annual HbA1c Screening']
        }
    },
    'kidney_disease': {
        'positive': {
            'is_positive': True,
            'title': 'Renal Functional Markers Abnormal - CKD Risk',
            'severity': 'High Clinical Risk - Nephrology Evaluation Required',
            'badge': 'ATTENTION REQUIRED',
            'status_color': 'danger',
            'description': 'AI multi-parametric analysis of urinalysis and blood biochemistry detected markers consistent with impaired glomerular filtration rate and chronic kidney disease (CKD).',
            'specialist': 'Consult a Board-Certified Nephrologist',
            'urgency_window': 'Urgent nephrology evaluation recommended within 48 to 72 hours.',
            'risk_factors': [
                'Proteinuria (Albumin present in urine reflecting glomerular leakage)',
                'Elevated Serum Creatinine (> 1.2 mg/dL) and Blood Urea Nitrogen (BUN)',
                'Abnormal urine specific gravity and microscopic sediment indicators (RBCs / Pus cells)',
                'Hypertension causing renal arteriolar nephrosclerosis',
                'Anemia or depleted hemoglobin associated with reduced erythropoietin secretion'
            ],
            'emergency_symptoms': [
                'Severe bilateral lower-extremity peripheral edema (swollen legs/ankles) or facial puffiness',
                'Marked reduction in urine output (oliguria/anuria) or foamy/cola-colored blood in urine',
                'Sudden severe shortness of breath due to pulmonary fluid overload',
                'Severe intractable nausea, metallic taste in mouth, cognitive fog, or muscle twitching (uremia signs)'
            ],
            'recommendations': [
                'Undergo immediate confirmatory Serum Creatinine, eGFR calculation, and 24-hour Urine Protein test',
                'Schedule a Renal Ultrasound (KUB - Kidney, Ureter, Bladder) to assess kidney size and cortical thickness',
                'Strictly avoid all Over-The-Counter Nephrotoxic drugs: Ibuprofen, Naproxen, Diclofenac (NSAIDs)',
                'Undergo strict blood pressure management aiming for systolic BP < 120-130 mm Hg',
                'Work with a renal dietician to regulate dietary protein, sodium, potassium, and phosphorus'
            ],
            'doctor_questions': [
                'What is my estimated Glomerular Filtration Rate (eGFR) and what stage of CKD does it correspond to?',
                'Are ACE inhibitors (like Enalapril) or ARBs (like Losartan) suitable to protect my renal microvasculature?',
                'Do I need to restrict dietary potassium (bananas, potatoes) or dietary phosphorus (dairy)?',
                'Are there reversible etiologies (such as renal stone obstruction, medication effect, or dehydration)?'
            ],
            'diet_dos': [
                'Low-sodium intake: keep sodium under 1,500 mg daily to reduce renal glomerular pressure',
                'Moderate, high-biological-value protein intake (strictly adhere to nephrologist-prescribed grams/day)',
                'Consume kidney-friendly fruits (apples, berries, cauliflower) based on serum potassium levels'
            ],
            'diet_donts': [
                'Strictly avoid excessive salt, processed canned goods, bouillon cubes, and MSG',
                'Avoid protein powders, high-dosage creatine supplements, and crash high-protein keto diets',
                'Avoid excessive dark colas and packaged foods with added phosphate preservatives'
            ],
            'recommended_tests': ['Serum Creatinine & eGFR', 'Blood Urea Nitrogen (BUN)', 'Urine Albumin-to-Creatinine Ratio (uACR)', 'Electrolyte Panel (K+, Na+, Ca2+)', 'Renal Ultrasound']
        },
        'negative': {
            'is_positive': False,
            'title': 'Renal Biomarkers Within Healthy Limits',
            'severity': 'Low Risk Detected - Healthy Kidney Function',
            'badge': 'OPTIMAL HEALTH',
            'status_color': 'safe',
            'description': 'Renal filtration, urinary parameters, serum creatinine, and electrolyte values indicate intact glomerular filtration and healthy renal physiology.',
            'specialist': 'General Practitioner / Family Physician',
            'urgency_window': 'Standard annual wellness checkup.',
            'tips': [
                'Drink adequate water throughout the day (approx. 2 to 2.5 liters) to support renal waste clearance',
                'Avoid habitual misuse of NSAID painkillers (e.g. ibuprofen); choose safer alternatives when needed',
                'Maintain healthy blood pressure and blood sugar, the two primary safeguards against renal damage',
                'Perform an annual routine urinalysis and basic metabolic panel'
            ],
            'diet_dos': [
                'Stay well-hydrated, consume fresh hydrating vegetables, and season foods with fresh herbs instead of excess salt'
            ],
            'diet_donts': [
                'Avoid excessive table salt and chronic dehydration during intense heat or exercise'
            ],
            'recommended_tests': ['Annual Routine Urinalysis', 'Annual Serum Creatinine Screening']
        }
    },
    'liver_disease': {
        'positive': {
            'is_positive': True,
            'title': 'Hepatic Enzyme & Function Indicators Abnormal',
            'severity': 'High Probability of Hepatic Dysfunction / Inflammation',
            'badge': 'ATTENTION REQUIRED',
            'status_color': 'danger',
            'description': 'Elevated transaminases (ALT/AST), alkaline phosphatase, or bilirubin ratios reflect potential hepatocellular injury, cholestasis, fatty liver changes, or hepatic inflammation.',
            'specialist': 'Consult a Hepatologist / Gastroenterologist',
            'urgency_window': 'Schedule a gastroenterologist/hepatologist visit within 3 to 5 days.',
            'risk_factors': [
                'Elevated Direct and Total Bilirubin indicating impaired hepatic clearance',
                'Elevated ALT (Alamine Aminotransferase) and AST indicating active hepatocyte necrosis',
                'Alkaline Phosphatase elevation suggesting biliary duct involvement or inflammation',
                'Altered Albumin and Albumin-to-Globulin (A/G) ratio indicating compromised protein synthesis',
                'Risk factors including metabolic syndrome, steatosis, viral exposure, or alcohol/toxin intake'
            ],
            'emergency_symptoms': [
                'Yellowing of skin or sclera of the eyes (jaundice) accompanied by dark amber urine and pale stools',
                'Abdominal distension or fluid retention in belly (ascites) and swelling in legs',
                'Vomiting blood or dark black tarry stools (melena - signs of esophageal variceal bleeding)',
                'Acute confusion, marked disorientation, slurred speech, or flapping hand tremors (hepatic encephalopathy)'
            ],
            'recommendations': [
                'Schedule a liver ultrasound or FibroScan to evaluate liver stiffness and steatosis stage',
                'Complete viral hepatitis serology panel (HBsAg for Hepatitis B, Anti-HCV for Hepatitis C)',
                'Strictly avoid 100% of alcoholic beverages and all non-essential hepatotoxic supplements/medications',
                'Review every regular prescription with your physician to ensure liver-safe dosing',
                'Undergo coagulation profile screening (Prothrombin Time / INR) to evaluate synthesis function'
            ],
            'doctor_questions': [
                'What is the underlying cause: metabolic dysfunction-associated steatohepatitis (MASH/NASH), viral, or toxic?',
                'Does my liver imaging show any signs of fibrosis or early portal hypertension?',
                'Are my current medications safe for my liver enzymes, or do dosages need adjustments?',
                'What lifestyle and weight loss targets will best reverse hepatic fat infiltration?'
            ],
            'diet_dos': [
                'Emphasize antioxidant-rich foods: cruciferous vegetables, green tea, black coffee (shown to protect hepatocytes)',
                'Anti-inflammatory Mediterranean diet with abundant omega-3s, artichokes, and fiber',
                'High-fiber plant-based proteins (beans, lentils, tofu) and lean poultry'
            ],
            'diet_donts': [
                'Strictly zero alcohol consumption - even small amounts accelerate ongoing hepatocyte damage',
                'Avoid high-fructose corn syrup, candy, sugary sodas (fructose drives hepatic de novo lipogenesis)',
                'Avoid saturated fats, fried foods, raw or undercooked shellfish (risk of Vibrio vulnificus)'
            ],
            'recommended_tests': ['Comprehensive Liver Function Panel (LFT)', 'Liver Ultrasound / FibroScan', 'Hepatitis B & C Serology', 'Coagulation PT/INR', 'Serum Ferritin']
        },
        'negative': {
            'is_positive': False,
            'title': 'Hepatic Function Within Normal Range',
            'severity': 'Low Risk Detected - Healthy Liver Profile',
            'badge': 'OPTIMAL HEALTH',
            'status_color': 'safe',
            'description': 'Serum bilirubin, alkaline phosphatase, transaminases, and protein synthesis parameters are balanced and indicative of normal hepatic metabolic function.',
            'specialist': 'Primary Care Physician',
            'urgency_window': 'Routine annual wellness screening.',
            'tips': [
                'Maintain a healthy weight to prevent metabolic liver infiltration and steatosis',
                'Avoid excessive or binge alcohol consumption to safeguard liver longevity',
                'Ensure Hepatitis A and B vaccinations are up-to-date',
                'Always use prescription and over-the-counter medications only according to directions'
            ],
            'diet_dos': [
                'Stay hydrated, eat fresh seasonal vegetables, and enjoy moderate black coffee for hepatic health'
            ],
            'diet_donts': [
                'Avoid heavy alcohol consumption, excessive fried foods, and untested herbal supplements'
            ],
            'recommended_tests': ['Annual Routine Liver Function Screening']
        }
    },
    'breast_cancer': {
        'malignant': {
            'is_positive': True,
            'title': 'Cellular Metrics Consistent with Malignancy',
            'severity': 'High Priority - Immediate Oncology Referral Required',
            'badge': 'ATTENTION REQUIRED',
            'status_color': 'danger',
            'description': 'AI feature vector analysis of cellular nucleus morphology (mean radius, texture irregularity, perimeter distortion, area expansion, and edge smoothness) matches characteristics of malignant breast neoplasms.',
            'specialist': 'Consult a Surgical Oncologist & Breast Oncology Specialist',
            'urgency_window': 'Urgent diagnostic confirmation within 24 to 48 hours.',
            'risk_factors': [
                'Increased nuclear mean radius and enlarged cellular area indicating rapid mitotic proliferation',
                'Nuclear border irregularity, high texture variance, and abnormal perimeter ratios',
                'Loss of smooth cellular boundary architecture typical of invasive neoplastic transformation',
                'Associated clinical factors: palpable lump, family history (BRCA1/BRCA2), or prior abnormal imaging'
            ],
            'emergency_symptoms': [
                'Hard, non-tender, immovable solitary breast mass or axillary lymph node enlargement',
                'Skin changes: peau d\'orange (orange-peel skin dimpling), breast skin retraction, or ulceration',
                'Spontaneous serosanguinous (bloody or clear pink) nipple discharge from a single duct',
                'Inverted nipple or newly developed erythema/warmth across the breast tissue'
            ],
            'recommendations': [
                'Consult an oncology center immediately for confirmatory clinical breast examination',
                'Schedule a Diagnostic Digital Bilateral Mammogram and high-resolution Breast Ultrasound',
                'Undergo a Core Needle Biopsy (CNB) of the suspicious lesion with histopathological review',
                'Request immunohistochemical staining (ER, PR, HER2-neu, and Ki-67 proliferation index)',
                'Prepare all prior imaging disks and clinical notes for the multidisciplinary tumor board review'
            ],
            'doctor_questions': [
                'What is the recommended biopsy method (ultrasound-guided core biopsy or stereotactic)?',
                'Do you recommend a Breast MRI to evaluate lesion multicentricity or contralateral tissue?',
                'Should I consult a genetic counselor for BRCA1, BRCA2, or multi-gene panel testing?',
                'What is the staging and anticipated timeline for treatment planning?'
            ],
            'diet_dos': [
                'Nutrient-dense whole-food diet: cruciferous vegetables (rich in sulforaphane), berries, flaxseed',
                'Adequate lean protein to maintain immune reserves and muscle mass during upcoming clinical workup',
                'Plant-based anti-inflammatory foods and hydration'
            ],
            'diet_donts': [
                'Strictly avoid alcohol consumption (alcohol increases circulating estrogens and oncogenic risk)',
                'Avoid ultra-processed meats, trans fats, and excess refined sugars',
                'Do not begin untested alternative supplements without oncologist clearance'
            ],
            'recommended_tests': ['Diagnostic Digital Mammogram', 'Breast Ultrasound', 'Image-Guided Core Needle Biopsy', 'Receptor Status (ER/PR/HER2)', 'Breast MRI']
        },
        'benign': {
            'is_positive': False,
            'title': 'Cellular Metrics Consistent with Benign Tissue',
            'severity': 'Low Risk Detected - Non-Malignant Characteristics',
            'badge': 'OPTIMAL HEALTH',
            'status_color': 'safe',
            'description': 'The nuclear dimensions, smoothness, and texture parameters align with benign, non-cancerous cellular structures (such as fibroadenoma or normal glandular tissue).',
            'specialist': 'Gynecologist / Women\'s Health Specialist',
            'urgency_window': 'Continue standard age-appropriate breast screening schedule.',
            'tips': [
                'Perform monthly breast self-examinations to stay familiar with your normal tissue texture',
                'Follow clinical guidelines for annual or biennial screening mammography (age 40+ or per doctor advice)',
                'Promptly report any new, persistent, or progressive palpable lumps to your physician',
                'Maintain a healthy body weight and active lifestyle'
            ],
            'diet_dos': [
                'Eat balanced, fiber-rich whole foods, antioxidant-rich fruits, and leafy greens'
            ],
            'diet_donts': [
                'Limit alcohol consumption and avoid smoking'
            ],
            'recommended_tests': ['Routine Screening Mammogram (as recommended by age)', 'Routine Annual Clinical Breast Exam']
        }
    },
    'malaria': {
        'infected': {
            'is_positive': True,
            'title': 'Parasitized Erythrocytes Detected (Malaria Positive)',
            'severity': 'Acute Parasitic Infection - Prompt Medical Treatment Mandated',
            'badge': 'ATTENTION REQUIRED',
            'status_color': 'danger',
            'description': 'Deep convolutional neural network image analysis of the Giemsa-stained thin blood smear detected intracellular trophozoite or ring-form inclusions consistent with Plasmodium parasite invasion.',
            'specialist': 'Consult an Infectious Disease Specialist / Emergency Physician',
            'urgency_window': 'Seek immediate medical evaluation today (within 6 to 12 hours).',
            'risk_factors': [
                'Recent travel to or residence in malaria-endemic tropical/subtropical regions',
                'Exposure to Anopheles mosquito bites without chemoprophylaxis or bed nets',
                'Immunosuppression, pregnancy, or lack of prior acquired immunity',
                'Delayed diagnosis risking progression to severe falciparum malaria complications'
            ],
            'emergency_symptoms': [
                'Paroxysmal shaking chills (rigors) followed by high spiking fever (103-105°F) and drenching sweats',
                'Altered mental status, confusion, seizures, or unresponsiveness (Cerebral Malaria warning!)',
                'Severe dark/black tea-colored urine (Blackwater fever due to massive intravascular hemolysis)',
                'Severe anemia, jaundice, persistent vomiting, or respiratory distress/acidotic breathing'
            ],
            'recommendations': [
                'Go to a clinic or hospital emergency room immediately for definitive blood smear confirmation and parasite density quantification',
                'Initiate prescribed Antimalarial therapy immediately (such as Artemisinin-based Combination Therapy - ACT)',
                'Maintain vigorous oral hydration with electrolyte solutions to support renal perfusion',
                'Do not discontinue antimalarial medication early; complete the exact full prescribed course',
                'Monitor for signs of hypoglycemia and severe hemolytic anemia'
            ],
            'doctor_questions': [
                'Which Plasmodium species is present (Plasmodium falciparum, vivax, ovale, or malariae)?',
                'What is the quantitative parasite count / percentage of infected red blood cells?',
                'If P. vivax or P. ovale, will Primaquine therapy be needed to eradicate dormant liver hypnozoites (after G6PD testing)?',
                'Do my platelet counts and hemoglobin levels require close in-hospital monitoring?'
            ],
            'diet_dos': [
                'High-electrolyte fluids: coconut water, oral rehydration salts (ORS), clear broths to replace sweat losses',
                'Easily digestible, nutrient-dense soft foods: rice congee, boiled potatoes, bananas, and yogurt',
                'Iron and vitamin C-rich foods during recovery to assist erythrocyte replenishment'
            ],
            'diet_donts': [
                'Avoid heavy, greasy, fried, or spicy foods that irritate a fever-stressed gastrointestinal tract',
                'Avoid caffeine, alcohol, and unpasteurized drinks',
                'Do not take unverified herbal remedies in place of verified antimalarial medication'
            ],
            'recommended_tests': ['Thick and Thin Blood Smears with Parasite Count', 'Malaria Rapid Diagnostic Test (RDT)', 'Complete Blood Count (CBC) with Platelets', 'Serum Bilirubin & Liver Panel', 'G6PD Enzyme Assay']
        },
        'uninfected': {
            'is_positive': False,
            'title': 'No Intra-Erythrocytic Parasites Detected',
            'severity': 'Negative for Malaria Parasites in Sample',
            'badge': 'OPTIMAL HEALTH',
            'status_color': 'safe',
            'description': 'Deep learning vision analysis found clean, normal erythrocyte morphology with no identifiable Plasmodium ring stages, schizonts, or trophozoites in the submitted blood smear image.',
            'specialist': 'General Practitioner / Travel Medicine Physician',
            'urgency_window': 'If active unexplained fever persists, follow up for repeat blood smear.',
            'tips': [
                'If you have an active fever after tropical travel, note that repeat blood smears over 48 hours may be needed as parasite levels fluctuate',
                'Continue mosquito bite prevention: DEET-based repellents, permethrin-treated clothing, bed nets',
                'Consider prescribed antimalarial prophylaxis if currently traveling in high-transmission areas',
                'Seek prompt medical evaluation if high fevers or rigors reoccur'
            ],
            'diet_dos': [
                'Maintain regular hydration, balanced nutrition, and adequate rest'
            ],
            'diet_donts': [
                'Avoid unprotected exposure to mosquitoes during dusk-to-dawn peak biting hours'
            ],
            'recommended_tests': ['Repeat Peripheral Blood Smear in 24-48 hours if febrile symptoms persist']
        }
    },
    'pneumonia': {
        'positive': {
            'is_positive': True,
            'title': 'Pulmonary Infiltrates / Consolidation Detected',
            'severity': 'Acute Lower Respiratory Infection - Clinical Action Required',
            'badge': 'ATTENTION REQUIRED',
            'status_color': 'danger',
            'description': 'Deep neural network chest radiograph visual analyzer detected radiographic opacification, alveolar consolidation, or interstitial infiltrates characteristic of bacterial or viral pneumonia.',
            'specialist': 'Consult a Pulmonologist / Respiratory Physician or Visit Urgent Care',
            'urgency_window': 'Obtain medical evaluation today (within 12 to 24 hours).',
            'risk_factors': [
                'Airspace consolidation or patchy opacity obscuring normal lung parenchyma on X-ray',
                'Recent viral upper respiratory infection (Influenza, RSV, COVID-19) predisposing to secondary pneumonia',
                'Age > 65 or underlying comorbidities (COPD, asthma, diabetes, heart failure, immunocompromise)',
                'History of smoking, vaping, or environmental pollutant inhalation'
            ],
            'emergency_symptoms': [
                'Severe shortness of breath or rapid labored breathing even while resting',
                'Pulse oximeter reading below 92% SpO2 (hypoxemia!) or blue/gray discoloration of lips/fingernails (cyanosis)',
                'Sharp stabbing pleuritic chest pain exacerbated by deep breathing or coughing',
                'Confusion, delirium, or severe lethargy, especially in older adults'
            ],
            'recommendations': [
                'Seek immediate clinical consultation for chest auscultation (stethoscope crackles/rhonchi evaluation)',
                'Have your blood oxygen saturation (SpO2) and respiratory rate measured',
                'Obtain sputum culture and Complete Blood Count (CBC) with differential to distinguish bacterial vs viral etiology',
                'Take all prescribed antibiotics or antivirals exactly on schedule; do not miss or discontinue doses early',
                'Practice deep breathing exercises or use an incentive spirometer to prevent alveolar collapse (atelectasis)'
            ],
            'doctor_questions': [
                'Is my pneumonia radiographic pattern bacterial (lobar) or viral/atypical (interstitial)?',
                'What is my CURB-65 or Pneumonia Severity Index (PSI) score, and is outpatient management safe for me?',
                'Do I require supplemental oxygen or nebulizer bronchodilator therapy?',
                'When should I schedule a follow-up chest radiograph to confirm complete resolution of infiltrates?'
            ],
            'diet_dos': [
                'Continuous warm hydration: broths, herbal teas with honey, warm water to thin mucus and ease expectoration',
                'Anti-inflammatory foods: garlic, turmeric, citrus fruits rich in Vitamin C, and clean proteins',
                'Small, frequent nutrient-dense meals that do not demand heavy respiratory effort to digest'
            ],
            'diet_donts': [
                'Strictly avoid active cigarette smoking, vaping, and secondhand smoke exposure',
                'Avoid cold dairy or ultra-thick sweet drinks if they worsen mucus sensation',
                'Avoid alcohol which impairs respiratory immune defense and dehydrates bronchial mucosa'
            ],
            'recommended_tests': ['Official Radiologist Chest X-Ray Review', 'Pulse Oximetry / Arterial Blood Gas', 'Complete Blood Count (CBC)', 'Sputum Gram Stain & Culture', 'Influenza & Respiratory Viral Panel']
        },
        'negative': {
            'is_positive': False,
            'title': 'Clear Lung Fields - No Consolidation Detected',
            'severity': 'Low Risk Detected - Clear Chest Radiograph',
            'badge': 'OPTIMAL HEALTH',
            'status_color': 'safe',
            'description': 'Deep learning radiograph visual inspection shows clear bilateral lung fields, sharp costophrenic angles, and normal bronchovascular markings without evidence of acute pneumonia consolidation.',
            'specialist': 'Primary Care Physician / General Practitioner',
            'urgency_window': 'If cough or fever persists, consult doctor for non-pneumonia conditions.',
            'tips': [
                'Maintain respiratory hygiene: regular hand washing and avoiding close contact with sick individuals',
                'Stay current with seasonal respiratory immunizations (Flu vaccine, Pneumococcal conjugate vaccine, COVID booster)',
                'Protect your lungs by staying in smoke-free environments and using masks during air quality alerts',
                'If symptoms like fever or persistent cough develop, consult a physician for clinical assessment'
            ],
            'diet_dos': [
                'Maintain good daily hydration, colorful antioxidant-rich vegetables, and clean whole foods'
            ],
            'diet_donts': [
                'Avoid tobacco products and occupational inhalant exposure'
            ],
            'recommended_tests': ['Routine Clinical Respiratory Auscultation as needed']
        }
    }
}

# ============================================================================
# WEB ROUTES
# ============================================================================

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# --- Heart Disease ---
@app.route('/heart', methods=['GET', 'POST'])
def heart():
    if request.method == 'POST':
        try:
            raw_data = {
                'age': float(request.form['age']),
                'sex': float(request.form['sex']),
                'cp': float(request.form['cp']),
                'bp': float(request.form['bp']),
                'chol': float(request.form['chol']),
                'fbs': float(request.form['fbs']),
                'ekg': float(request.form['ekg']),
                'max_hr': float(request.form['max_hr']),
                'exang': float(request.form['exang']),
                'oldpeak': float(request.form['oldpeak']),
                'slope': float(request.form['slope']),
                'ca': float(request.form['ca']),
                'thal': float(request.form['thal'])
            }

            features_vec = [
                raw_data['age'], raw_data['sex'], raw_data['cp'], raw_data['bp'],
                raw_data['chol'], raw_data['fbs'], raw_data['ekg'], raw_data['max_hr'],
                raw_data['exang'], raw_data['oldpeak'], raw_data['slope'], raw_data['ca'],
                raw_data['thal']
            ]
            
            prediction = heart_model.predict([features_vec])[0]
            is_pos = (prediction == 1)
            result = "Presence of Cardiovascular Risk" if is_pos else "Absence of Significant Risk"
            insight_key = 'positive' if is_pos else 'negative'
            insights = DISEASE_INSIGHTS['heart_disease'][insight_key]

            # Formatted patient data for printable report
            patient_summary = [
                {'label': 'Age', 'value': f"{int(raw_data['age'])} years"},
                {'label': 'Sex', 'value': 'Male' if raw_data['sex'] == 1 else 'Female'},
                {'label': 'Chest Pain Type', 'value': ['Typical Angina', 'Atypical Angina', 'Non-Anginal', 'Asymptomatic'][int(raw_data['cp'])]},
                {'label': 'Resting Blood Pressure', 'value': f"{int(raw_data['bp'])} mm Hg"},
                {'label': 'Serum Cholesterol', 'value': f"{int(raw_data['chol'])} mg/dL"},
                {'label': 'Fasting Blood Sugar > 120', 'value': 'Yes' if raw_data['fbs'] == 1 else 'No'},
                {'label': 'Resting ECG', 'value': ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'][int(raw_data['ekg'])]},
                {'label': 'Max Heart Rate', 'value': f"{int(raw_data['max_hr'])} bpm"},
                {'label': 'Exercise Induced Angina', 'value': 'Yes' if raw_data['exang'] == 1 else 'No'},
                {'label': 'ST Depression (Oldpeak)', 'value': f"{raw_data['oldpeak']}"},
                {'label': 'ST Slope', 'value': ['Upsloping', 'Flat', 'Downsloping'][int(raw_data['slope'])]},
                {'label': 'Major Vessels Fluro', 'value': f"{int(raw_data['ca'])}"},
                {'label': 'Thallium Stress Result', 'value': {3: 'Normal', 6: 'Fixed Defect', 7: 'Reversible Defect'}.get(int(raw_data['thal']), str(raw_data['thal']))}
            ]
            
            return render_template('heart.html', prediction=result, insights=insights, is_positive=is_pos, patient_summary=patient_summary)
        except Exception as e:
            return render_template('heart.html', error=str(e))
    return render_template('heart.html')

# --- Diabetes ---
@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        try:
            gender = request.form['gender']
            smoking = request.form['smoking_history']
            age = float(request.form['age'])
            hypertension = int(request.form['hypertension'])
            heart_disease = int(request.form['heart_disease'])
            bmi = float(request.form['bmi'])
            HbA1c_level = float(request.form['HbA1c_level'])
            blood_glucose_level = float(request.form['blood_glucose_level'])

            data = {
                'age': age,
                'hypertension': hypertension,
                'heart_disease': heart_disease,
                'bmi': bmi,
                'HbA1c_level': HbA1c_level,
                'blood_glucose_level': blood_glucose_level,
                'gender_Male': 1 if gender == 'Male' else 0,
                'gender_Other': 1 if gender == 'Other' else 0,
                'smoking_history_current': 1 if smoking == 'current' else 0,
                'smoking_history_ever': 1 if smoking == 'ever' else 0,
                'smoking_history_former': 1 if smoking == 'former' else 0,
                'smoking_history_never': 1 if smoking == 'never' else 0,
                'smoking_history_not current': 1 if smoking == 'not current' else 0
            }
            
            feature_order = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 
                             'gender_Male', 'gender_Other', 'smoking_history_current', 'smoking_history_ever', 
                             'smoking_history_former', 'smoking_history_never', 'smoking_history_not current']
            
            feature_vector = [data[col] for col in feature_order]
            prediction = diabetes_model.predict([feature_vector])[0]
            is_pos = (prediction == 1)
            result = "Positive - Elevated Diabetes Risk" if is_pos else "Negative - Normal Glycemic Indicators"
            insight_key = 'positive' if is_pos else 'negative'
            insights = DISEASE_INSIGHTS['diabetes'][insight_key]

            patient_summary = [
                {'label': 'Gender', 'value': gender},
                {'label': 'Age', 'value': f"{int(age)} years"},
                {'label': 'Hypertension History', 'value': 'Yes' if hypertension == 1 else 'No'},
                {'label': 'Heart Disease History', 'value': 'Yes' if heart_disease == 1 else 'No'},
                {'label': 'Smoking History', 'value': smoking.capitalize()},
                {'label': 'Body Mass Index (BMI)', 'value': f"{bmi:.1f} kg/m²"},
                {'label': 'HbA1c Level', 'value': f"{HbA1c_level:.1f}%"},
                {'label': 'Blood Glucose Level', 'value': f"{int(blood_glucose_level)} mg/dL"}
            ]
            
            return render_template('diabetes.html', prediction=result, insights=insights, is_positive=is_pos, patient_summary=patient_summary)
        except Exception as e:
            return render_template('diabetes.html', error=str(e))
    return render_template('diabetes.html')

# --- Kidney Disease ---
@app.route('/kidney', methods=['GET', 'POST'])
def kidney():
    if request.method == 'POST':
        try:
            field_keys = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 
                          'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 
                          'cad', 'appet', 'pe', 'ane']
            
            features = [float(request.form[k]) for k in field_keys]
            prediction = kidney_model.predict([features])[0]
            is_pos = (prediction == 1)
            result = "Positive - Chronic Kidney Disease Indicators" if is_pos else "Negative - Healthy Renal Function"
            insight_key = 'positive' if is_pos else 'negative'
            insights = DISEASE_INSIGHTS['kidney_disease'][insight_key]

            patient_summary = [
                {'label': 'Age', 'value': f"{int(features[0])} years"},
                {'label': 'Blood Pressure', 'value': f"{int(features[1])} mm Hg"},
                {'label': 'Specific Gravity', 'value': f"{features[2]:.3f}"},
                {'label': 'Albumin (Urine)', 'value': f"Grade {int(features[3])}"},
                {'label': 'Sugar (Urine)', 'value': f"Grade {int(features[4])}"},
                {'label': 'Red Blood Cells', 'value': 'Normal' if features[5] == 1 else 'Abnormal'},
                {'label': 'Pus Cells', 'value': 'Normal' if features[6] == 1 else 'Abnormal'},
                {'label': 'Blood Glucose Random', 'value': f"{int(features[9])} mg/dL"},
                {'label': 'Blood Urea', 'value': f"{features[10]} mg/dL"},
                {'label': 'Serum Creatinine', 'value': f"{features[11]} mg/dL"},
                {'label': 'Hemoglobin', 'value': f"{features[14]} g/dL"},
                {'label': 'Packed Cell Volume', 'value': f"{int(features[15])}%"},
                {'label': 'Hypertension', 'value': 'Yes' if features[18] == 1 else 'No'},
                {'label': 'Diabetes Mellitus', 'value': 'Yes' if features[19] == 1 else 'No'}
            ]
            
            return render_template('kidney.html', prediction=result, insights=insights, is_positive=is_pos, patient_summary=patient_summary)
        except Exception as e:
            return render_template('kidney.html', error=str(e))
    return render_template('kidney.html')

# --- Liver Disease ---
@app.route('/liver', methods=['GET', 'POST'])
def liver():
    if request.method == 'POST':
        try:
            gender_val = 1 if request.form['Gender'] == 'Male' else 0
            features = [
                float(request.form['Age']),
                gender_val,
                float(request.form['Total_Bilirubin']),
                float(request.form['Direct_Bilirubin']),
                float(request.form['Alkaline_Phosphotase']),
                float(request.form['Alamine_Aminotransferase']),
                float(request.form['Aspartate_Aminotransferase']),
                float(request.form['Total_Protiens']),
                float(request.form['Albumin']),
                float(request.form['Albumin_and_Globulin_Ratio'])
            ]
            
            prediction = liver_model.predict([features])[0]
            is_pos = (prediction == 1)
            result = "High Probability of Liver Disease" if is_pos else "Low Probability of Liver Disease"
            insight_key = 'positive' if is_pos else 'negative'
            insights = DISEASE_INSIGHTS['liver_disease'][insight_key]

            patient_summary = [
                {'label': 'Age', 'value': f"{int(features[0])} years"},
                {'label': 'Gender', 'value': 'Male' if gender_val == 1 else 'Female'},
                {'label': 'Total Bilirubin', 'value': f"{features[2]:.1f} mg/dL"},
                {'label': 'Direct Bilirubin', 'value': f"{features[3]:.1f} mg/dL"},
                {'label': 'Alkaline Phosphatase (ALP)', 'value': f"{int(features[4])} U/L"},
                {'label': 'Alamine Aminotransferase (ALT)', 'value': f"{int(features[5])} U/L"},
                {'label': 'Aspartate Aminotransferase (AST)', 'value': f"{int(features[6])} U/L"},
                {'label': 'Total Proteins', 'value': f"{features[7]:.1f} g/dL"},
                {'label': 'Albumin', 'value': f"{features[8]:.1f} g/dL"},
                {'label': 'A/G Ratio', 'value': f"{features[9]:.2f}"}
            ]
            
            return render_template('liver.html', prediction=result, insights=insights, is_positive=is_pos, patient_summary=patient_summary)
        except Exception as e:
            return render_template('liver.html', error=str(e))
    return render_template('liver.html')

# --- Breast Cancer ---
@app.route('/breast_cancer', methods=['GET', 'POST'])
def breast_cancer():
    if request.method == 'POST':
        try:
            features = [
                float(request.form['mean_radius']),
                float(request.form['mean_texture']),
                float(request.form['mean_perimeter']),
                float(request.form['mean_area']),
                float(request.form['mean_smoothness'])
            ]
            
            prediction = breast_cancer_model.predict([features])[0]
            # 0 = Malignant, 1 = Benign
            is_pos = (prediction == 0)
            result = "Malignant Neoplasm Indicators" if is_pos else "Benign Cellular Characteristics"
            insight_key = 'malignant' if is_pos else 'benign'
            insights = DISEASE_INSIGHTS['breast_cancer'][insight_key]

            patient_summary = [
                {'label': 'Mean Nuclear Radius', 'value': f"{features[0]:.2f} µm"},
                {'label': 'Mean Texture Variation', 'value': f"{features[1]:.2f}"},
                {'label': 'Mean Perimeter', 'value': f"{features[2]:.2f} µm"},
                {'label': 'Mean Area', 'value': f"{features[3]:.1f} µm²"},
                {'label': 'Mean Smoothness Index', 'value': f"{features[4]:.4f}"}
            ]
            
            return render_template('breast_cancer.html', prediction=result, insights=insights, is_positive=is_pos, patient_summary=patient_summary)
        except Exception as e:
            return render_template('breast_cancer.html', error=str(e))
    return render_template('breast_cancer.html')

# --- Malaria ---
@app.route('/malaria', methods=['GET', 'POST'])
def malaria():
    if request.method == 'POST':
        try:
            # Check for sample trigger or file upload
            sample_type = request.form.get('sample_type')
            filepath = None
            filename = None

            if sample_type == 'infected':
                filepath = os.path.join(app.config['SAMPLES_FOLDER'], 'malaria_infected.png')
                filename = 'malaria_infected.png'
                uploaded_url = url_for('static', filename='samples/malaria_infected.png')
            elif sample_type == 'uninfected':
                filepath = os.path.join(app.config['SAMPLES_FOLDER'], 'malaria_uninfected.png')
                filename = 'malaria_uninfected.png'
                uploaded_url = url_for('static', filename='samples/malaria_uninfected.png')
            else:
                if 'file' not in request.files:
                    return render_template('malaria.html', error="No file selected")
                file = request.files['file']
                if file.filename == '':
                    return render_template('malaria.html', error="No selected file")
                
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_url = url_for('static', filename='uploads/' + filename)

            if malaria_model is None:
                return render_template('malaria.html', error="Malaria model is not loaded")

            img = keras_image.load_img(filepath, target_size=(50, 50))
            img_array = keras_image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            pred = malaria_model.predict(img_array, verbose=0)
            pred_class = np.argmax(pred[0])  # 0 = Uninfected, 1 = Parasitized
            confidence = float(pred[0][pred_class]) * 100

            is_pos = (pred_class == 1)
            result = f"Infected (Parasitized) - {confidence:.1f}% Confidence" if is_pos else f"Uninfected (Healthy) - {confidence:.1f}% Confidence"
            insight_key = 'infected' if is_pos else 'uninfected'
            insights = DISEASE_INSIGHTS['malaria'][insight_key]

            patient_summary = [
                {'label': 'Image Analyzed', 'value': filename},
                {'label': 'Resolution Tested', 'value': '50 x 50 pixels (Deep CNN Microscopic Analysis)'},
                {'label': 'Detection Classification', 'value': 'Parasitized (Infected)' if is_pos else 'Uninfected (Negative)'},
                {'label': 'AI Confidence Score', 'value': f"{confidence:.1f}%"}
            ]

            return render_template('malaria.html', prediction=result, insights=insights, is_positive=is_pos, 
                                   patient_summary=patient_summary, uploaded_image_url=uploaded_url)
        except Exception as e:
            return render_template('malaria.html', error=str(e))
    return render_template('malaria.html')

# --- Pneumonia ---
@app.route('/pneumonia', methods=['GET', 'POST'])
def pneumonia():
    if request.method == 'POST':
        try:
            sample_type = request.form.get('sample_type')
            filepath = None
            filename = None

            if sample_type == 'detected':
                filepath = os.path.join(app.config['SAMPLES_FOLDER'], 'pneumonia_detected.jpeg')
                filename = 'pneumonia_detected.jpeg'
                uploaded_url = url_for('static', filename='samples/pneumonia_detected.jpeg')
            else:
                if 'file' not in request.files:
                    return render_template('pneumonia.html', error="No file selected")
                file = request.files['file']
                if file.filename == '':
                    return render_template('pneumonia.html', error="No selected file")
                
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_url = url_for('static', filename='uploads/' + filename)

            if pneumonia_model is None:
                return render_template('pneumonia.html', error="Pneumonia model is not loaded")

            img = keras_image.load_img(filepath, target_size=(150, 150))
            img_array = keras_image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            pred = pneumonia_model.predict(img_array, verbose=0)
            prob = float(pred[0][0])
            is_pos = (prob > 0.5)
            confidence = (prob if is_pos else (1.0 - prob)) * 100

            result = f"Pneumonia Detected - {confidence:.1f}% Confidence" if is_pos else f"Normal Lung Fields - {confidence:.1f}% Confidence"
            insight_key = 'positive' if is_pos else 'negative'
            insights = DISEASE_INSIGHTS['pneumonia'][insight_key]

            patient_summary = [
                {'label': 'Chest X-Ray File', 'value': filename},
                {'label': 'Matrix Dimension', 'value': '150 x 150 Deep Convolutional Layer Scan'},
                {'label': 'Pulmonary Assessment', 'value': 'Consolidation / Infiltrates Present' if is_pos else 'Clear / Unremarkable'},
                {'label': 'AI Confidence Level', 'value': f"{confidence:.1f}%"}
            ]

            return render_template('pneumonia.html', prediction=result, insights=insights, is_positive=is_pos,
                                   patient_summary=patient_summary, uploaded_image_url=uploaded_url)
        except Exception as e:
            return render_template('pneumonia.html', error=str(e))
    return render_template('pneumonia.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

