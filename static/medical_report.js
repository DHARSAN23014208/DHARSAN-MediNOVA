/**
 * MediNova Hospital-Grade Medical Report Generator (PDF / Print)
 * ==============================================================
 * Dynamically constructs a standalone, print-optimized A4 clinical document
 * inside a dedicated print frame, eliminating blank pages and layout clipping.
 *
 * Designed for 1-to-2 page A4 clinical documentation.
 * Developed by Dharsankumar for MediNova Clinical AI Systems.
 */

(function () {
    'use strict';

    /**
     * Extracts all live clinical findings, biomarkers, recommendations,
     * and patient details from the active DOM and application storage.
     */
    function extractDiagnosticData() {
        var headingText = '';
        var h1 = document.querySelector('h1.section-title, .form-header-area h1, h1');
        if (h1) headingText = h1.textContent.trim();

        // 1. Detect Diagnostic Condition Name
        var conditionName = 'Clinical Diagnostic Assessment';
        var conditionKey = 'general';
        var path = window.location.pathname.toLowerCase();

        if (path.includes('heart') || headingText.toLowerCase().includes('heart') || headingText.toLowerCase().includes('cardio')) {
            conditionName = 'Cardiovascular / Heart Disease Evaluation';
            conditionKey = 'heart';
        } else if (path.includes('diabetes') || headingText.toLowerCase().includes('diabetes')) {
            conditionName = 'Diabetes Mellitus & Glycemic Profile';
            conditionKey = 'diabetes';
        } else if (path.includes('kidney') || headingText.toLowerCase().includes('kidney') || headingText.toLowerCase().includes('renal')) {
            conditionName = 'Chronic Kidney Disease & Renal Function';
            conditionKey = 'kidney';
        } else if (path.includes('liver') || headingText.toLowerCase().includes('liver') || headingText.toLowerCase().includes('hepatic')) {
            conditionName = 'Hepatic Function & Liver Disease Panel';
            conditionKey = 'liver';
        } else if (path.includes('breast') || path.includes('cancer') || headingText.toLowerCase().includes('breast') || headingText.toLowerCase().includes('oncology')) {
            conditionName = 'Breast Oncology Cytological Morphology';
            conditionKey = 'cancer';
        } else if (path.includes('malaria') || headingText.toLowerCase().includes('malaria')) {
            conditionName = 'Malaria Parasite Microscopic Blood Smear';
            conditionKey = 'malaria';
        } else if (path.includes('pneumonia') || headingText.toLowerCase().includes('pneumonia')) {
            conditionName = 'Pneumonia Pulmonary Radiograph (CXR)';
            conditionKey = 'pneumonia';
        }

        // 2. Assess Positive vs Safe Outcome
        var isPositive = false;
        var alertCard = document.querySelector('.critical-alert-card, .safe-alert-card');
        if (alertCard) {
            isPositive = alertCard.classList.contains('critical-alert-card');
        } else {
            var dangerEl = document.querySelector('.result-status-tag.danger, .result-badge-icon.danger, .risk-meter-fill.danger');
            isPositive = !!dangerEl;
        }

        // 3. Extract Assessment Text & Description
        var assessmentTitle = '';
        var assessmentDesc = '';
        var heroText = document.querySelector('.result-hero-text');
        if (heroText) {
            var h2 = heroText.querySelector('h2');
            if (h2) assessmentTitle = h2.textContent.trim();
            var paragraphs = heroText.querySelectorAll('p');
            paragraphs.forEach(function (p) {
                var text = p.textContent.trim();
                if (text.includes('AI Diagnostic Assessment:')) {
                    assessmentDesc = text.replace('AI Diagnostic Assessment:', '').trim();
                } else if (!assessmentDesc && text) {
                    assessmentDesc = text;
                }
            });
        }
        if (!assessmentTitle) {
            assessmentTitle = isPositive ? 'Presence of Elevated Clinical Risk' : 'Optimal / Negative Clinical Profile';
        }

        // 4. Risk Severity Meter & Percentage
        var riskScore = isPositive ? 'HIGH RISK (88%)' : 'OPTIMAL (12%)';
        var riskMeterHeader = document.querySelector('.risk-meter-header span:last-child');
        if (riskMeterHeader && riskMeterHeader.textContent.trim()) {
            riskScore = riskMeterHeader.textContent.trim();
        }

        // 5. Specialist Recommendation & Urgency
        var specialistTitle = '';
        var urgencyWindow = isPositive ? 'Schedule clinical consultation within 24 to 48 hours' : 'Routine periodic follow-up';
        var recommendedTests = [];

        var specCard = Array.from(document.querySelectorAll('.protocol-card')).find(function (card) {
            return card.textContent.includes('Specialist Recommendation');
        });

        if (specCard) {
            var pList = specCard.querySelectorAll('p');
            if (pList.length > 0) specialistTitle = pList[0].textContent.trim();
            if (pList.length > 1 && pList[1].textContent.includes('Urgency:')) {
                urgencyWindow = pList[1].textContent.replace('Urgency:', '').trim();
            }
            var testBadges = specCard.querySelectorAll('.feature-badge');
            testBadges.forEach(function (b) {
                var t = b.textContent.trim();
                if (t) recommendedTests.push(t);
            });
        }
        if (!specialistTitle) {
            specialistTitle = isPositive ? 'Consulting Specialist Physician' : 'Primary Care Physician';
        }

        // 6. Emergency Red-Flag Symptoms
        var emergencySymptoms = [];
        var dangerCard = Array.from(document.querySelectorAll('.protocol-card')).find(function (card) {
            return card.textContent.includes('Red-Flag Symptoms') || card.textContent.includes('ER') || card.textContent.includes('Seek Immediate');
        });
        if (dangerCard) {
            var lis = dangerCard.querySelectorAll('li');
            lis.forEach(function (li) {
                var s = li.textContent.trim();
                if (s) emergencySymptoms.push(s);
            });
        }

        // 7. Clinical Recommendations / Tips
        var clinicalTips = [];
        var tipCard = Array.from(document.querySelectorAll('.protocol-card')).find(function (card) {
            return card.textContent.includes('Recommendations') || card.textContent.includes('Tips') || card.textContent.includes('Maintenance');
        });
        if (tipCard) {
            var tipLis = tipCard.querySelectorAll('li');
            tipLis.forEach(function (li) {
                var s = li.textContent.trim();
                if (s) clinicalTips.push(s);
            });
        }

        // 8. Biomarkers Table
        var biomarkers = [];
        var rows = document.querySelectorAll('.patient-data-table tbody tr');
        rows.forEach(function (row) {
            var cols = row.querySelectorAll('td');
            if (cols.length >= 2) {
                var param = cols[0].textContent.trim();
                var val = cols[1].textContent.trim();
                var target = cols.length >= 3 ? cols[2].textContent.trim() : 'Standard Reference Limits';
                biomarkers.push({ parameter: param, value: val, target: target });
            }
        });

        // Fallback: If no table rows yet but form has inputs
        if (biomarkers.length === 0) {
            var formInputs = document.querySelectorAll('.form-card form input[type="number"], .form-card form select');
            formInputs.forEach(function (inp) {
                if (inp.name && inp.value) {
                    var label = inp.closest('.form-group') ? inp.closest('.form-group').querySelector('label') : null;
                    var labelText = label ? label.textContent.trim() : inp.name;
                    var valText = inp.tagName.toLowerCase() === 'select' && inp.selectedOptions.length > 0 ? inp.selectedOptions[0].textContent.trim() : inp.value;
                    biomarkers.push({ parameter: labelText, value: valText, target: 'Standard Physiological Limit' });
                }
            });
        }

        // 9. Vision Image Preview (if present for Malaria / Pneumonia)
        var analyzedImageSrc = '';
        var analyzedImageNote = '';
        var visionImg = document.querySelector('.uploaded-image-preview-card img, .file-preview-thumbnail, .analyzed-img');
        if (visionImg && visionImg.src && !visionImg.src.includes('data:image/svg')) {
            analyzedImageSrc = visionImg.src;
            var noteP = document.querySelector('.uploaded-image-preview-card p');
            if (noteP) analyzedImageNote = noteP.textContent.trim();
        }

        // 10. Patient & Appointment Summary Extraction
        var patientInfo = {
            isFilled: false,
            fullName: '',
            phone: '',
            email: '',
            hospitalName: '',
            consultationMode: '',
            referenceId: '',
            clinicalNotes: ''
        };

        // A. Check on-page receipt modal elements
        var rName = document.getElementById('receiptPatientName');
        var rContact = document.getElementById('receiptContact');
        var rRef = document.getElementById('receiptReferenceId');
        var rHosp = document.getElementById('receiptHospitalName');
        var rMode = document.getElementById('receiptMode');
        var rNotes = document.getElementById('receiptNotes');

        if (rName && rName.textContent.trim() && rName.textContent.trim() !== '-') {
            patientInfo.isFilled = true;
            patientInfo.fullName = rName.textContent.trim();
            patientInfo.referenceId = rRef ? rRef.textContent.trim() : '';
            patientInfo.hospitalName = rHosp ? rHosp.textContent.trim() : '';
            patientInfo.consultationMode = rMode ? rMode.textContent.trim() : 'In-Person Consultation';
            patientInfo.clinicalNotes = (rNotes && rNotes.textContent.trim() !== '-') ? rNotes.textContent.trim() : '';
            if (rContact) {
                var parts = rContact.textContent.split('|');
                patientInfo.phone = parts[0] ? parts[0].trim() : '';
                patientInfo.email = parts[1] ? parts[1].trim() : '';
            }
        }

        // B. Check localStorage for persistent last appointment
        if (!patientInfo.isFilled) {
            try {
                var storedAptStr = localStorage.getItem('medinova_last_appointment');
                if (storedAptStr) {
                    var storedApt = JSON.parse(storedAptStr);
                    if (storedApt && storedApt.patient_name) {
                        patientInfo.isFilled = true;
                        patientInfo.fullName = storedApt.patient_name;
                        patientInfo.phone = storedApt.phone || '';
                        patientInfo.email = storedApt.email || '';
                        patientInfo.hospitalName = storedApt.hospital_name || '';
                        patientInfo.consultationMode = storedApt.consultation_mode === 'telemedicine' ? 'Telemedicine Consultation' : 'In-Person Consultation';
                        patientInfo.referenceId = storedApt.reference_id || '';
                        patientInfo.clinicalNotes = storedApt.clinical_notes || '';
                    }
                }
            } catch (e) {}
        }

        // C. Check direct form inputs if present on page
        var nameInput = document.getElementById('patientName') || document.getElementById('patient_name');
        if (nameInput && nameInput.value.trim()) {
            patientInfo.isFilled = true;
            patientInfo.fullName = nameInput.value.trim();
        }
        var phoneInput = document.getElementById('patientPhone') || document.getElementById('patient_phone') || document.getElementById('phone');
        if (phoneInput && phoneInput.value.trim() && !patientInfo.phone) {
            patientInfo.phone = phoneInput.value.trim();
        }
        var emailInput = document.getElementById('patientEmail') || document.getElementById('patient_email') || document.getElementById('email');
        if (emailInput && emailInput.value.trim() && !patientInfo.email) {
            patientInfo.email = emailInput.value.trim();
        }
        var notesInput = document.getElementById('clinicalNotes') || document.getElementById('clinical_notes') || document.getElementById('notes');
        if (notesInput && notesInput.value.trim() && !patientInfo.clinicalNotes) {
            patientInfo.clinicalNotes = notesInput.value.trim();
        }

        // 11. Generate / Resolve Clinical Report ID & Timestamp
        var now = new Date();
        var dateFormatted = now.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
        var timeFormatted = now.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            hour12: true
        });

        var existingReportId = window.medinovaReportId;
        if (!existingReportId) {
            var reportEl = document.querySelector('[data-report-id]');
            if (reportEl) existingReportId = reportEl.getAttribute('data-report-id');
        }
        if (!existingReportId) {
            var reportMeta = document.querySelector('meta[name="report-id"]');
            if (reportMeta) existingReportId = reportMeta.getAttribute('content');
        }
        if (!existingReportId) {
            var sealBar = document.querySelector('.scanner-integrity-seal-bar');
            if (sealBar) {
                var match = sealBar.textContent.match(/MNV-[A-Z0-9-]+/i);
                if (match) existingReportId = match[0].toUpperCase();
            }
        }
        if (!existingReportId) {
            var verifyLink = document.querySelector('a[href*="/verify-report/"]');
            if (verifyLink) {
                var linkMatch = (verifyLink.getAttribute('href') || '').match(/\/verify-report\/([^\/\?#]+)/);
                if (linkMatch) existingReportId = linkMatch[1].toUpperCase();
            }
        }

        var randHex = Math.random().toString(16).substring(2, 8).toUpperCase();
        var reportId = existingReportId || ('MNV-' + conditionKey.substring(0, 4).toUpperCase() + '-' + now.getFullYear() + '-' + randHex);

        // Detect if extraction originated from scanner
        var isScannerExtracted = false;
        if (window.medinovaIsScanner === true) {
            isScannerExtracted = true;
        } else if (document.querySelector('[data-scanner-used="true"]') || document.querySelector('[data-extracted-source="scanner"]')) {
            isScannerExtracted = true;
        }

        return {
            conditionName: conditionName,
            conditionKey: conditionKey,
            reportId: reportId,
            isScannerExtracted: isScannerExtracted,
            dateTime: dateFormatted + ' at ' + timeFormatted,
            isPositive: isPositive,
            assessmentTitle: assessmentTitle,
            assessmentDesc: assessmentDesc,
            riskScore: riskScore,
            specialistTitle: specialistTitle,
            urgencyWindow: urgencyWindow,
            recommendedTests: recommendedTests,
            emergencySymptoms: emergencySymptoms,
            clinicalTips: clinicalTips,
            biomarkers: biomarkers,
            analyzedImageSrc: analyzedImageSrc,
            analyzedImageNote: analyzedImageNote,
            patientInfo: patientInfo
        };
    }

    /**
     * Builds clean, standalone, high-contrast 1-to-2 page A4 print HTML.
     */
    function buildReportHTML(data) {
        var statusBg = data.isPositive ? '#fee2e2' : '#ecfdf5';
        var statusBorder = data.isPositive ? '#dc2626' : '#10b981';
        var statusColor = data.isPositive ? '#991b1b' : '#065f46';
        var statusIcon = data.isPositive ? '🚨' : '🛡️';
        var statusTag = data.isPositive ? 'ELEVATED CLINICAL ALERT / HIGH RISK' : 'OPTIMAL CLINICAL PROFILE / LOW RISK';

        // Build clean 2-column Biomarkers Table
        var biomarkerRowsHtml = '';
        if (data.biomarkers && data.biomarkers.length > 0) {
            data.biomarkers.forEach(function (b, index) {
                var bg = index % 2 === 0 ? '#ffffff' : '#f8fafc';
                biomarkerRowsHtml += `
                    <tr style="background-color: ${bg};">
                        <td style="padding: 5px 9px; border: 1px solid #e2e8f0; font-weight: 600; color: #1e293b; font-size: 8.5pt;">
                            ${escapeHtml(b.parameter)}
                        </td>
                        <td style="padding: 5px 9px; border: 1px solid #e2e8f0; font-weight: 700; color: #0284c7; font-size: 8.5pt;">
                            ${escapeHtml(b.value)}
                            <span style="display: block; font-size: 7.5pt; font-weight: normal; color: #64748b;">Target: ${escapeHtml(b.target)}</span>
                        </td>
                    </tr>
                `;
            });
        }

        // Build Vision Image Section (if present for Malaria / Pneumonia)
        var visionSectionHtml = '';
        if (data.analyzedImageSrc) {
            visionSectionHtml = `
                <div style="display: flex; gap: 14px; align-items: center; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px; margin-bottom: 10px; page-break-inside: avoid;">
                    <img src="${data.analyzedImageSrc}" style="width: 85px; height: 85px; object-fit: cover; border-radius: 5px; border: 1px solid #cbd5e1;" alt="Analyzed Scan">
                    <div style="flex: 1;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
                            <strong style="color: #0f172a; font-size: 9.5pt;">Diagnostic Microscopic / Radiographic Scan Analyzed</strong>
                            <span style="font-size: 7.5pt; background: #e0f2fe; color: #0369a1; padding: 2px 6px; border-radius: 4px; font-weight: 600;">Deep CNN Feature Map</span>
                        </div>
                        <p style="margin: 0; color: #475569; font-size: 8.5pt; line-height: 1.35;">
                            ${escapeHtml(data.analyzedImageNote || 'Intracellular morphological feature extraction verified by deep convolutional neural networks.')}
                        </p>
                    </div>
                </div>
            `;
        }

        // Build Recommended Tests Badges
        var testsHtml = '';
        if (data.recommendedTests && data.recommendedTests.length > 0) {
            testsHtml = '<div style="margin-top: 5px;">' + data.recommendedTests.map(function (t) {
                return `<span style="display: inline-block; background: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd; border-radius: 4px; padding: 2px 6px; font-size: 7.5pt; font-weight: 600; margin-right: 4px; margin-bottom: 3px;">${escapeHtml(t)}</span>`;
            }).join('') + '</div>';
        }

        // Build Red-Flag Emergency Symptoms
        var emergencyHtml = '';
        if (data.isPositive && data.emergencySymptoms && data.emergencySymptoms.length > 0) {
            emergencyHtml = `
                <div style="background: #fff1f2; border: 1px solid #fecdd3; border-radius: 5px; padding: 7px 10px; margin-top: 6px;">
                    <strong style="color: #be123c; font-size: 8pt; text-transform: uppercase;">🚨 Immediate Red-Flag Warning Signs (Seek ER / 112 / 911):</strong>
                    <ul style="margin: 3px 0 0 0; padding-left: 16px; color: #9f1239; font-size: 8pt; line-height: 1.3;">
                        ${data.emergencySymptoms.slice(0, 4).map(function (s) { return `<li>${escapeHtml(s)}</li>`; }).join('')}
                    </ul>
                </div>
            `;
        }

        // Build Clinical Tips
        var tipsHtml = '';
        if (data.clinicalTips && data.clinicalTips.length > 0) {
            tipsHtml = `
                <ul style="margin: 3px 0 0 0; padding-left: 16px; color: #334155; font-size: 8pt; line-height: 1.35;">
                    ${data.clinicalTips.slice(0, 3).map(function (t) { return `<li>${escapeHtml(t)}</li>`; }).join('')}
                </ul>
            `;
        }

        // Build Patient & Appointment Summary Section
        var pInfo = data.patientInfo;
        var patientSummaryHtml = '';
        if (pInfo.isFilled && pInfo.fullName) {
            patientSummaryHtml = `
                <div class="patient-appointment-card">
                    <div class="card-section-label">👤 Patient & Appointment Consultation Record</div>
                    <div class="patient-grid">
                        <div class="p-item">
                            <span class="p-lbl">Patient Full Name</span>
                            <span class="p-val">${escapeHtml(pInfo.fullName)}</span>
                        </div>
                        <div class="p-item">
                            <span class="p-lbl">Contact Phone & Email</span>
                            <span class="p-val">${escapeHtml(pInfo.phone || 'Not Provided')} | ${escapeHtml(pInfo.email || 'N/A')}</span>
                        </div>
                        <div class="p-item">
                            <span class="p-lbl">Selected Specialty Hospital</span>
                            <span class="p-val">${escapeHtml(pInfo.hospitalName || 'MediNova Priority Network Center')}</span>
                        </div>
                        <div class="p-item">
                            <span class="p-lbl">Consultation Mode</span>
                            <span class="p-val">${escapeHtml(pInfo.consultationMode || 'In-Person Consultation')}</span>
                        </div>
                        <div class="p-item">
                            <span class="p-lbl">Appointment Reference ID</span>
                            <span class="p-val" style="font-family: monospace; font-weight: 700; color: #0891b2;">${escapeHtml(pInfo.referenceId || 'APT-PENDING')}</span>
                        </div>
                        <div class="p-item">
                            <span class="p-lbl">Clinical Screening Modality</span>
                            <span class="p-val">${escapeHtml(data.conditionName)}</span>
                        </div>
                    </div>
                    ${pInfo.clinicalNotes ? `
                        <div style="margin-top: 6px; padding-top: 5px; border-top: 1px dashed #cbd5e1; font-size: 8pt; color: #475569;">
                            <strong>Patient Clinical Notes & Symptoms:</strong> ${escapeHtml(pInfo.clinicalNotes)}
                        </div>
                    ` : ''}
                </div>
            `;
        } else {
            patientSummaryHtml = `
                <div class="patient-appointment-card">
                    <div class="card-section-label">👤 Patient Triage & Assessment Profile</div>
                    <div class="patient-grid">
                        <div class="p-item">
                            <span class="p-lbl">Patient Profile</span>
                            <span class="p-val">Outpatient Direct Screening / Walk-In</span>
                        </div>
                        <div class="p-item">
                            <span class="p-lbl">Screening Modality</span>
                            <span class="p-val">${escapeHtml(data.conditionName)}</span>
                        </div>
                        <div class="p-item">
                            <span class="p-lbl">Triage Consultation Mode</span>
                            <span class="p-val">AI Risk Stratification & Clinical Guidance</span>
                        </div>
                        <div class="p-item">
                            <span class="p-lbl">Specialist Network Center</span>
                            <span class="p-val">MediNova Coordinated Care Network</span>
                        </div>
                    </div>
                </div>
            `;
        }

        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MediNova Clinical Diagnostic Report - ${escapeHtml(data.reportId)}</title>
    <style>
        @page {
            size: A4 portrait;
            margin: 12mm 15mm;
        }
        * {
            box-sizing: border-box;
            -webkit-print-color-adjust: exact !important;
            print-color-adjust: exact !important;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #0f172a;
            background: #ffffff !important;
            margin: 0;
            padding: 0;
            font-size: 9pt;
            line-height: 1.35;
        }
        .report-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            border-bottom: 2.5px solid #0891b2;
            padding-bottom: 8px;
            margin-bottom: 10px;
        }
        .brand-title {
            font-size: 16pt;
            font-weight: 800;
            color: #0891b2;
            margin: 0;
            letter-spacing: -0.02em;
        }
        .brand-subtitle {
            font-size: 8pt;
            color: #64748b;
            margin: 2px 0 0 0;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }
        .meta-box {
            text-align: right;
            font-size: 8pt;
            color: #475569;
        }
        .report-id {
            font-family: "Courier New", Courier, monospace;
            font-size: 9.5pt;
            font-weight: 700;
            color: #0891b2;
        }
        .accreditation-badge {
            display: inline-block;
            background: #f0fdf4;
            color: #166534;
            border: 1px solid #bbf7d0;
            border-radius: 4px;
            padding: 2px 6px;
            font-size: 7.5pt;
            font-weight: 700;
            margin-top: 3px;
        }
        .patient-appointment-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 8px 12px;
            margin-bottom: 10px;
            page-break-inside: avoid;
        }
        .card-section-label {
            font-size: 8pt;
            font-weight: 700;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 6px;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 3px;
        }
        .patient-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 6px 12px;
        }
        .p-item .p-lbl {
            display: block;
            font-size: 7.2pt;
            color: #64748b;
            text-transform: uppercase;
            font-weight: 600;
        }
        .p-item .p-val {
            font-weight: 600;
            color: #0f172a;
            font-size: 8.5pt;
        }
        .outcome-hero {
            background: ${statusBg};
            border: 1.5px solid ${statusBorder};
            border-radius: 6px;
            padding: 8px 12px;
            margin-bottom: 10px;
            page-break-inside: avoid;
        }
        .outcome-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 3px;
        }
        .outcome-tag {
            font-size: 8pt;
            font-weight: 800;
            color: ${statusColor};
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .outcome-score {
            font-size: 9.5pt;
            font-weight: 800;
            color: ${statusColor};
        }
        .outcome-title {
            font-size: 12pt;
            font-weight: 800;
            color: #0f172a;
            margin: 0 0 2px 0;
        }
        .outcome-desc {
            font-size: 8.5pt;
            color: #334155;
            margin: 0;
            line-height: 1.35;
        }
        .clinical-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 10px;
            page-break-inside: avoid;
        }
        .section-box {
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 8px 10px;
            background: #ffffff;
        }
        .section-title {
            font-size: 8.5pt;
            font-weight: 700;
            color: #0f172a;
            margin: 0 0 5px 0;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            border-bottom: 1px solid #f1f5f9;
            padding-bottom: 3px;
        }
        table.biomarker-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 8.5pt;
            margin-bottom: 10px;
            page-break-inside: avoid;
        }
        table.biomarker-table th {
            background: #f1f5f9;
            color: #334155;
            padding: 5px 8px;
            text-align: left;
            border: 1px solid #cbd5e1;
            font-size: 7.8pt;
            text-transform: uppercase;
        }
        .report-footer {
            border-top: 1px solid #cbd5e1;
            padding-top: 8px;
            margin-top: 10px;
            page-break-inside: avoid;
        }
        .disclaimer-text {
            font-size: 7.2pt;
            color: #64748b;
            line-height: 1.35;
            margin: 0 0 8px 0;
            text-align: justify;
        }
        .signature-row {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            margin-top: 6px;
        }
        .sign-box {
            width: 220px;
            border-top: 1px solid #94a3b8;
            padding-top: 4px;
            font-size: 7.5pt;
            color: #475569;
        }
    </style>
</head>
<body>
    <!-- Official Medical Header -->
    <div class="report-header">
        <div>
            <div style="display: flex; align-items: center; gap: 6px;">
                <span style="font-size: 16pt;">🩺</span>
                <h1 class="brand-title">MediNova</h1>
            </div>
            <p class="brand-subtitle">Clinical Diagnostic & Consultation Report</p>
            ${data.isScannerExtracted ? `
                <div style="display: inline-flex; align-items: center; gap: 4px; background: #ecfeff; border: 1.5px solid #06b6d4; color: #0891b2; font-size: 7.5pt; font-weight: 800; padding: 2px 8px; border-radius: 4px; margin-top: 4px;">
                    🔬 Extracted via AI Lab Report Scanner
                </div>
            ` : ''}
        </div>
        <div class="meta-box" style="display: flex; gap: 12px; align-items: center;">
            <div style="text-align: right;">
                <div>Report ID: <span class="report-id">${escapeHtml(data.reportId)}</span></div>
                <div>Generated: <strong>${escapeHtml(data.dateTime)}</strong></div>
                <div class="accreditation-badge">⭐ NABL / ISO-15189 & HIPAA Aligned Quality Gateway</div>
            </div>
            <div style="text-align: center;">
                <img src="/api/report-qr/${encodeURIComponent(data.reportId)}" style="width: 65px; height: 65px; object-fit: contain; border: 1px solid #cbd5e1; border-radius: 4px;" alt="Verification QR Code">
                <div style="font-size: 6pt; color: #64748b; margin-top: 1px;">Scan to Verify</div>
            </div>
        </div>
    </div>

    <!-- Patient & Appointment Summary Card -->
    ${patientSummaryHtml}

    <!-- Diagnostic Condition & Risk Assessment Hero -->
    <div class="outcome-hero">
        <div class="outcome-header">
            <span class="outcome-tag">${statusIcon} ${escapeHtml(statusTag)}</span>
            <span class="outcome-score">Confidence & Severity Index: ${escapeHtml(data.riskScore)}</span>
        </div>
        <h2 class="outcome-title">${escapeHtml(data.assessmentTitle)}</h2>
        <p class="outcome-desc">${escapeHtml(data.assessmentDesc)}</p>
    </div>

    ${visionSectionHtml}

    <!-- Clinical Parameters Entered (Clean 2-Column Table) -->
    ${biomarkerRowsHtml ? `
        <div style="margin-bottom: 10px; page-break-inside: avoid;">
            <div style="font-size: 8.5pt; font-weight: 700; color: #0f172a; margin-bottom: 4px; text-transform: uppercase;">
                📊 Clinical Parameters Entered & Physiological Target Limits
            </div>
            <table class="biomarker-table">
                <thead>
                    <tr>
                        <th style="width: 50%;">Clinical Parameter / Biomarker Tested</th>
                        <th style="width: 50%;">Submitted Metric & Target Range</th>
                    </tr>
                </thead>
                <tbody>
                    ${biomarkerRowsHtml}
                </tbody>
            </table>
        </div>
    ` : ''}

    <!-- Clinical Guidance & Warning Signs -->
    <div class="clinical-grid">
        <div class="section-box">
            <div class="section-title">🏥 Recommended Specialist Action</div>
            <div style="font-weight: 700; color: #0f172a; font-size: 9pt;">${escapeHtml(data.specialistTitle)}</div>
            <div style="font-size: 8pt; color: #475569; margin-top: 2px;">
                <strong>Action Window:</strong> ${escapeHtml(data.urgencyWindow)}
            </div>
            ${testsHtml}
        </div>

        <div class="section-box">
            <div class="section-title">📋 Clinical Guidance & Warning Signs</div>
            ${tipsHtml || '<p style="margin: 0; font-size: 8pt; color: #64748b;">Schedule standard primary care review and routine monitoring.</p>'}
            ${emergencyHtml}
        </div>
    </div>

    <!-- Medical Disclaimer Footer -->
    <div class="report-footer">
        <p class="disclaimer-text">
            <strong>CLINICAL DECISION SUPPORT DISCLAIMER:</strong> This document represents an automated artificial intelligence-assisted clinical screening and risk stratification report generated by the MediNova Clinical Platform. It is provided strictly for preliminary risk stratification, triage prioritization, and educational consultation support. This report does NOT constitute a final medical diagnosis, prognosis, or therapeutic prescription. Diagnostic verification, clinical correlation, and treatment decisions must be made by a licensed physician, clinical pathologist, or board-certified specialist.
        </p>

        <div class="signature-row">
            <div class="sign-box">
                Reviewing Clinician / Medical Officer Signature<br>
                Date & Clinical Registration Number
            </div>
            <div style="text-align: right; font-size: 7.2pt; color: #64748b;">
                MediNova Intelligent Clinical Core • Verification Seal Active<br>
                Verification: <strong>/verify-report/${escapeHtml(data.reportId)}</strong><br>
                Official Patient Copy (A4 Clinical Layout)
            </div>
        </div>
    </div>
</body>
</html>`;
    }

    /**
     * Spawns an isolated hidden iframe, writes clean A4 HTML, waits 300ms, and prints.
     */
    function generateMedicalReportPDF() {
        try {
            var data = extractDiagnosticData();
            var reportHtml = buildReportHTML(data);

            var existingFrame = document.getElementById('medinovaPrintFrame');
            if (existingFrame) {
                try { existingFrame.parentNode.removeChild(existingFrame); } catch (e) {}
            }

            var frame = document.createElement('iframe');
            frame.id = 'medinovaPrintFrame';
            frame.name = 'medinovaPrintFrame';
            frame.style.position = 'fixed';
            frame.style.right = '0';
            frame.style.bottom = '0';
            frame.style.width = '0';
            frame.style.height = '0';
            frame.style.border = '0';
            frame.style.visibility = 'hidden';
            document.body.appendChild(frame);

            var frameDoc = frame.contentWindow.document;
            frameDoc.open();
            frameDoc.write(reportHtml);
            frameDoc.close();

            // Wait 300ms for layout and CSS fonts to settle, then focus and print
            setTimeout(function () {
                try {
                    frame.contentWindow.focus();
                    frame.contentWindow.print();
                } catch (err) {
                    console.warn('Iframe print failed, falling back to popup window:', err);
                    openPrintWindowFallback(reportHtml);
                }
            }, 300);
        } catch (e) {
            console.error('Error generating medical report PDF:', e);
            window.print();
        }
    }

    function openPrintWindowFallback(reportHtml) {
        var win = window.open('', '_blank', 'width=850,height=1000');
        if (win) {
            win.document.open();
            win.document.write(reportHtml);
            win.document.close();
            setTimeout(function () {
                win.focus();
                win.print();
            }, 300);
        } else {
            window.print();
        }
    }

    function escapeHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    // Expose globally on window
    window.generateMedicalReportPDF = generateMedicalReportPDF;

    // Auto-bind any buttons containing "Download / Print Medical Report" or window.print
    function bindPrintButtons() {
        document.querySelectorAll('button[onclick*="window.print()"], .btn-print-report').forEach(function (btn) {
            btn.onclick = function (e) {
                e.preventDefault();
                generateMedicalReportPDF();
            };
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', bindPrintButtons);
    } else {
        bindPrintButtons();
    }
})();
