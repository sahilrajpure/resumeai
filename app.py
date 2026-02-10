import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import pickle
import PyPDF2
import io
import re
from datetime import datetime
from zoneinfo import ZoneInfo

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="HR Resume Screening System",
    page_icon="👔",
    layout="wide"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .winner-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin: 20px 0;
    }
    .runner-up-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        margin: 15px 0;
    }
    .candidate-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .requirement-box {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 10px 0;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 4px solid #667eea;
    }
    .strength-badge {
        display: inline-block;
        padding: 5px 15px;
        background: #28a745;
        color: white;
        border-radius: 20px;
        margin: 5px;
        font-weight: bold;
    }
    .weakness-badge {
        display: inline-block;
        padding: 5px 15px;
        background: #dc3545;
        color: white;
        border-radius: 20px;
        margin: 5px;
        font-weight: bold;
    }
    .comparison-table {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL AND ARTIFACTS
# ============================================
@st.cache_resource
def load_model_and_artifacts():
    """Load trained model and preprocessing artifacts"""
    try:
        model = keras.models.load_model('resume_lstm_model.h5')
        
        with open('resume_tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        
        with open('resume_label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        with open('resume_config.pkl', 'rb') as f:
            config = pickle.load(f)
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'label_encoder': label_encoder,
            'config': config
        }
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# ============================================
# TEXT PROCESSING FUNCTIONS
# ============================================
def clean_resume_text(text):
    """Clean resume text"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d{10}|\(\d{3}\)\s?\d{3}-\d{4}', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = ' '.join(text.split())
    
    return text

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def extract_skills(text):
    """Extract technical skills from resume"""
    skills_keywords = {
        'Programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'go', 'rust', 'sql'],
        'Machine Learning': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'ml', 'ai', 'neural network'],
        'Web Development': ['html', 'css', 'react', 'angular', 'vue', 'nodejs', 'django', 'flask', 'frontend', 'backend'],
        'Data Science': ['data analysis', 'pandas', 'numpy', 'matplotlib', 'data science', 'statistics', 'tableau', 'power bi'],
        'Cloud & DevOps': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'ci/cd', 'devops'],
        'Database': ['mysql', 'postgresql', 'mongodb', 'oracle', 'redis', 'database'],
        'Mobile': ['android', 'ios', 'react native', 'flutter', 'mobile development'],
        'Management': ['team lead', 'project management', 'agile', 'scrum', 'leadership', 'manager']
    }
    
    text_lower = text.lower()
    found_skills = {}
    
    for category, keywords in skills_keywords.items():
        found = [skill for skill in keywords if skill in text_lower]
        if found:
            found_skills[category] = found
    
    return found_skills

def extract_experience(text):
    """Extract years of experience"""
    patterns = [
        r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
        r'experience[:\s]+(\d+)\+?\s*years?',
        r'(\d+)\+?\s*yrs?\s+experience'
    ]
    
    years = []
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        years.extend([int(y) for y in matches])
    
    return max(years) if years else 0

def extract_education(text):
    """Extract education level"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['phd', 'ph.d', 'doctorate']):
        return 'PhD/Doctorate'
    elif any(word in text_lower for word in ['master', 'msc', 'm.sc', 'mba', 'mtech', 'm.tech']):
        return 'Masters'
    elif any(word in text_lower for word in ['bachelor', 'btech', 'b.tech', 'bsc', 'b.sc', 'be', 'b.e']):
        return 'Bachelors'
    else:
        return 'Not Specified'

def calculate_requirement_match(resume_text, requirements):
    """Calculate how well resume matches job requirements"""
    resume_lower = resume_text.lower()
    
    matches = []
    for req in requirements:
        req_lower = req.lower()
        # Check if requirement keywords are in resume
        req_words = set(req_lower.split())
        resume_words = set(resume_lower.split())
        
        common = req_words.intersection(resume_words)
        match_score = (len(common) / len(req_words)) * 100 if req_words else 0
        
        matches.append({
            'requirement': req,
            'score': match_score,
            'matched': match_score > 30  # 30% threshold
        })
    
    avg_match = np.mean([m['score'] for m in matches]) if matches else 0
    
    return matches, avg_match

# ============================================
# PREDICT RESUME CATEGORY
# ============================================
def predict_resume(resume_text, artifacts, requirements=None):
    """Predict resume category and analyze against requirements"""
    
    # Clean text
    cleaned_text = clean_resume_text(resume_text)
    
    if not cleaned_text or len(cleaned_text) < 10:
        return None
    
    config = artifacts['config']
    max_length = config['max_length']
    
    # Tokenize and predict
    sequence = artifacts['tokenizer'].texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    
    prediction = artifacts['model'].predict(padded, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class] * 100
    
    category = artifacts['label_encoder'].classes_[predicted_class]
    
    # Extract additional features
    skills = extract_skills(resume_text)
    experience_years = extract_experience(resume_text)
    education = extract_education(resume_text)
    
    # Calculate requirement match
    requirement_matches = None
    requirement_score = 0
    
    if requirements:
        requirement_matches, requirement_score = calculate_requirement_match(
            resume_text, requirements
        )
    
    # Calculate final score
    # 40% AI confidence + 30% requirement match + 20% experience + 10% education
    edu_score = {'PhD/Doctorate': 100, 'Masters': 80, 'Bachelors': 60, 'Not Specified': 30}.get(education, 30)
    exp_score = min(experience_years * 10, 100)  # Cap at 100
    
    if requirements:
        final_score = (
            confidence * 0.4 +
            requirement_score * 0.3 +
            exp_score * 0.2 +
            edu_score * 0.1
        )
    else:
        final_score = (
            confidence * 0.6 +
            exp_score * 0.3 +
            edu_score * 0.1
        )
    
    return {
        'category': category,
        'confidence': confidence,
        'skills': skills,
        'experience_years': experience_years,
        'education': education,
        'requirement_matches': requirement_matches,
        'requirement_score': requirement_score,
        'final_score': final_score,
        'text_length': len(resume_text),
        'cleaned_text': cleaned_text
    }

# ============================================
# MAIN APPLICATION
# ============================================
def main():
    
    # Header
    st.markdown('<div class="main-header">👔 HR Resume Screening System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Best Candidate Selector for Hiring Teams</div>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner('🔄 Loading AI model...'):
        artifacts = load_model_and_artifacts()
    
    if artifacts is None:
        st.stop()
    
    st.success('✅ AI Model Ready!')
    
    # Initialize session state
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    
    # ============================================
    # STEP 1: DEFINE JOB REQUIREMENTS
    # ============================================
    st.header("📋 Step 1: Define Job Requirements")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Job Position Details")
        
        job_title = st.text_input(
            "Job Title *",
            placeholder="e.g., Senior Python Developer",
            help="Enter the position you're hiring for"
        )
        
        department = st.selectbox(
            "Department *",
            ["Technology", "Data Science", "Marketing", "Sales", "HR", "Finance", "Operations", "Other"]
        )
        
        experience_required = st.slider(
            "Years of Experience Required *",
            min_value=0,
            max_value=20,
            value=3,
            step=1
        )
        
        education_required = st.selectbox(
            "Minimum Education *",
            ["Bachelors", "Masters", "PhD/Doctorate", "Any"]
        )
    
    with col2:
        st.subheader("Quick Info")
        st.info("""
        📌 **Instructions:**
        
        1. Fill job details
        2. Add key requirements
        3. Upload 2-5 resumes
        4. Get best candidate
        
        ⏱️ Takes ~15 seconds
        """)
    
    st.subheader("Key Requirements")
    st.write("Add specific skills, qualifications, or requirements (one per line)")
    
    requirements_text = st.text_area(
        "Requirements *",
        height=150,
        placeholder="Example:\nPython programming\nMachine Learning experience\nTeam leadership\nAWS cloud\nSQL databases",
        help="Enter each requirement on a new line"
    )
    
    requirements = [req.strip() for req in requirements_text.split('\n') if req.strip()]
    
    if requirements:
        st.success(f"✅ {len(requirements)} requirements added")
        with st.expander("View Requirements"):
            for i, req in enumerate(requirements, 1):
                st.write(f"{i}. {req}")
    
    # ============================================
    # STEP 2: UPLOAD RESUMES
    # ============================================
    st.header("📤 Step 2: Upload Candidate Resumes")
    
    st.info("📁 Upload 2 to 5 resume PDFs for comparison")
    
    uploaded_files = st.file_uploader(
        "Upload Resume PDFs",
        type=['pdf'],
        accept_multiple_files=True,
        help="Select 2-5 PDF resumes"
    )
    
    # Validation
    can_analyze = False
    
    if not job_title:
        st.warning("⚠️ Please enter job title")
    elif not requirements:
        st.warning("⚠️ Please add at least one requirement")
    elif not uploaded_files:
        st.warning("⚠️ Please upload resume PDFs")
    elif len(uploaded_files) < 2:
        st.warning(f"⚠️ Please upload at least 3 resumes (currently: {len(uploaded_files)})")
    elif len(uploaded_files) > 5:
        st.warning(f"⚠️ Maximum 5 resumes allowed (currently: {len(uploaded_files)})")
    else:
        can_analyze = True
        st.success(f"✅ {len(uploaded_files)} resumes uploaded and ready for analysis")
    
    # ============================================
    # STEP 3: ANALYZE AND SELECT BEST CANDIDATE
    # ============================================
    if can_analyze:
        st.header("🔍 Step 3: Analyze & Select Best Candidate")
        
        if st.button("🚀 Analyze Resumes", type="primary", use_container_width=True):
            
            results = []
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process each resume
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Analyzing {uploaded_file.name}...")
                
                # Extract text
                resume_text = extract_text_from_pdf(uploaded_file)
                
                if resume_text:
                    # Analyze resume
                    analysis = predict_resume(resume_text, artifacts, requirements)
                    
                    if analysis:
                        results.append({
                            'filename': uploaded_file.name,
                            'analysis': analysis
                        })
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("✅ Analysis complete!")
            progress_bar.empty()
            
            if results:
                # Sort by final score
                results = sorted(results, key=lambda x: x['analysis']['final_score'], reverse=True)
                
                st.session_state.results = results
                st.session_state.analysis_done = True
                st.session_state.job_title = job_title
                st.session_state.requirements = requirements
    
    # ============================================
    # DISPLAY RESULTS
    # ============================================
    if st.session_state.get('analysis_done', False):
        
        results = st.session_state.results
        
        st.markdown("---")
        st.header("🏆 Analysis Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        winner = results[0]
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Total Candidates</h4>
                <h2>{len(results)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Winner Score</h4>
                <h2>{winner['analysis']['final_score']:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Category</h4>
                <h2>{winner['analysis']['category']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Experience</h4>
                <h2>{winner['analysis']['experience_years']} yrs</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # ============================================
        # BEST CANDIDATE (WINNER)
        # ============================================
        st.subheader("🥇 Best Candidate Recommendation")
        
        best = results[0]
        analysis = best['analysis']
        
        st.markdown(f"""
        <div class="winner-card">
            <h2>🏆 {best['filename']}</h2>
            <h3>Overall Match Score: {analysis['final_score']:.1f}/100</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed analysis of winner
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Key Metrics")
            st.write(f"**AI Confidence:** {analysis['confidence']:.1f}%")
            st.write(f"**Requirement Match:** {analysis['requirement_score']:.1f}%")
            st.write(f"**Category:** {analysis['category']}")
            st.write(f"**Experience:** {analysis['experience_years']} years")
            st.write(f"**Education:** {analysis['education']}")
            
            st.markdown("### ✅ Strengths")
            if analysis['skills']:
                for category, skills in analysis['skills'].items():
                    st.markdown(f"<span class='strength-badge'>{category}</span>", unsafe_allow_html=True)
            else:
                st.write("General technical background")
        
        with col2:
            st.markdown("### 📋 Requirement Analysis")
            
            if analysis['requirement_matches']:
                matched = [r for r in analysis['requirement_matches'] if r['matched']]
                not_matched = [r for r in analysis['requirement_matches'] if not r['matched']]
                
                st.write(f"**✅ Matched:** {len(matched)}/{len(analysis['requirement_matches'])}")
                
                for match in matched:
                    st.success(f"✓ {match['requirement']} ({match['score']:.0f}% match)")
                
                if not_matched:
                    st.write(f"**⚠️ Partially Matched/Missing:**")
                    for match in not_matched:
                        st.warning(f"⚠ {match['requirement']} ({match['score']:.0f}% match)")
        
        # Why this candidate is best
        st.markdown("### 💡 Why This Candidate?")
        
        reasons = []
        
        if analysis['final_score'] >= 80:
            reasons.append(f"✅ **Excellent overall match** ({analysis['final_score']:.1f}%) - significantly above average")
        
        if analysis['confidence'] >= 85:
            reasons.append(f"✅ **Strong AI confidence** ({analysis['confidence']:.1f}%) - resume closely matches trained patterns")
        
        if analysis['requirement_score'] >= 60:
            reasons.append(f"✅ **Good requirement alignment** ({analysis['requirement_score']:.1f}%) - meets most job requirements")
        
        if analysis['experience_years'] >= experience_required:
            reasons.append(f"✅ **Meets experience requirement** ({analysis['experience_years']} years vs {experience_required} required)")
        
        if analysis['skills']:
            skill_count = sum(len(s) for s in analysis['skills'].values())
            reasons.append(f"✅ **Diverse skill set** - {len(analysis['skills'])} categories, {skill_count} specific skills identified")
        
        if not reasons:
            reasons.append("✅ **Best among submitted candidates** based on comprehensive analysis")
        
        for reason in reasons:
            st.markdown(reason)
        
        # ============================================
        # OTHER CANDIDATES
        # ============================================
        if len(results) > 1:
            st.markdown("---")
            st.subheader("📊 Other Candidates")
            
            for idx, result in enumerate(results[1:], 2):
                analysis = result['analysis']
                
                card_class = "runner-up-card" if idx == 2 else "candidate-card"
                medal = "🥈" if idx == 2 else "🥉" if idx == 3 else f"#{idx}"
                
                with st.expander(f"{medal} {result['filename']} - Score: {analysis['final_score']:.1f}%"):
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Key Metrics:**")
                        st.write(f"• AI Confidence: {analysis['confidence']:.1f}%")
                        st.write(f"• Requirement Match: {analysis['requirement_score']:.1f}%")
                        st.write(f"• Experience: {analysis['experience_years']} years")
                        st.write(f"• Education: {analysis['education']}")
                        
                        if analysis['skills']:
                            st.write(f"\n**Skills Found:**")
                            for category, skills in analysis['skills'].items():
                                st.write(f"• {category}: {len(skills)} skills")
                    
                    with col2:
                        st.write("**Requirement Match:**")
                        if analysis['requirement_matches']:
                            matched = sum(1 for r in analysis['requirement_matches'] if r['matched'])
                            st.write(f"Matched: {matched}/{len(analysis['requirement_matches'])}")
                            
                            for match in analysis['requirement_matches'][:5]:  # Show top 5
                                icon = "✅" if match['matched'] else "⚠️"
                                st.write(f"{icon} {match['requirement']}: {match['score']:.0f}%")
        
        # ============================================
        # COMPARISON TABLE
        # ============================================
        st.markdown("---")
        st.subheader("📊 Side-by-Side Comparison")
        
        comparison_data = []
        for idx, result in enumerate(results, 1):
            analysis = result['analysis']
            comparison_data.append({
                'Rank': f"#{idx}",
                'Candidate': result['filename'],
                'Final Score': f"{analysis['final_score']:.1f}%",
                'AI Confidence': f"{analysis['confidence']:.1f}%",
                'Requirement Match': f"{analysis['requirement_score']:.1f}%",
                'Experience': f"{analysis['experience_years']} yrs",
                'Education': analysis['education'],
                'Category': analysis['category']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # ============================================
        # DOWNLOAD REPORTS
        # ============================================
        st.markdown("---")
        st.subheader("📥 Download Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Detailed report for winner
            winner_report = f"""
HIRING RECOMMENDATION REPORT
{'='*60}

Position: {st.session_state.job_title}
Department: {department}
Analysis Date: {datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")}

{'='*60}
RECOMMENDED CANDIDATE
{'='*60}

Candidate: {best['filename']}
Overall Match Score: {best['analysis']['final_score']:.1f}/100

RECOMMENDATION: STRONGLY RECOMMENDED FOR INTERVIEW

KEY METRICS:
-----------
• AI Confidence: {best['analysis']['confidence']:.1f}%
• Requirement Match: {best['analysis']['requirement_score']:.1f}%
• Category: {best['analysis']['category']}
• Experience: {best['analysis']['experience_years']} years
• Education: {best['analysis']['education']}

SKILLS IDENTIFIED:
-----------------
"""
            if best['analysis']['skills']:
                for category, skills in best['analysis']['skills'].items():
                    winner_report += f"\n{category}:\n"
                    for skill in skills:
                        winner_report += f"  • {skill}\n"
            
            winner_report += f"""
REQUIREMENT ANALYSIS:
--------------------
"""
            if best['analysis']['requirement_matches']:
                for match in best['analysis']['requirement_matches']:
                    status = "✓ MET" if match['matched'] else "⚠ PARTIAL"
                    winner_report += f"{status}: {match['requirement']} ({match['score']:.0f}% match)\n"
            
            winner_report += f"""
STRENGTHS:
---------
"""
            for reason in reasons:
                winner_report += f"{reason}\n"
            
            winner_report += f"""
{'='*60}
HIRING TEAM NOTES:
{'='*60}

[Space for interviewer notes]




{'='*60}
Report generated by AI Resume Screening System
"""
            
            st.download_button(
                label="📄 Download Winner Report",
                data=winner_report,
                file_name=f"recommended_candidate_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            # Complete comparison CSV
            csv = comparison_df.to_csv(index=False)
            
            st.download_button(
                label="📊 Download Complete Comparison (CSV)",
                data=csv,
                file_name=f"candidate_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Reset button
        st.markdown("---")
        if st.button("🔄 Start New Analysis", use_container_width=True):
            st.session_state.analysis_done = False
            st.rerun()

if __name__ == "__main__":

    main()





