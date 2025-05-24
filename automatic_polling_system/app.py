import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import uuid
from typing import List, Dict, Any
import re
from collections import Counter, defaultdict
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Automatic Polling System",
    # page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize ChromaDB client
@st.cache_resource
def init_chromadb():
    client = chromadb.Client(Settings(persist_directory="./chroma_db"))
    return client

# Initialize sentence transformer for embeddings
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

class TranscriptProcessor:
    def __init__(self, chroma_client, embedding_model):
        self.chroma_client = chroma_client
        self.embedding_model = embedding_model
        self.collection_name = "host_context"
        
    def setup_collection(self):
        """Setup ChromaDB collection for host context"""
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
        except:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def chunk_host_content(self, transcripts: List[Dict], host_name: str) -> List[Dict]:
        """Extract and chunk host's spoken content"""
        host_content = []
        
        for transcript in transcripts:
            if transcript['speaker'] == host_name:
                # Split long content into meaningful chunks
                content = transcript['text']
                sentences = re.split(r'(?<=[.!?])\s+', content)
                
                # Group sentences into chunks of 2-3 sentences for better context
                chunk_size = 3
                for i in range(0, len(sentences), chunk_size):
                    chunk = ' '.join(sentences[i:i+chunk_size])
                    if len(chunk.strip()) > 50:  # Only meaningful chunks
                        host_content.append({
                            'content': chunk,
                            'timestamp': transcript['timestamp'],
                            'chunk_id': str(uuid.uuid4())
                        })
        
        return host_content
    
    def store_context(self, host_content: List[Dict]):
        """Store host context in ChromaDB"""
        if not host_content:
            return
            
        contents = [chunk['content'] for chunk in host_content]
        embeddings = self.embedding_model.encode(contents).tolist()
        
        self.collection.add(
            documents=contents,
            embeddings=embeddings,
            metadatas=[{
                'timestamp': chunk['timestamp'],
                'chunk_id': chunk['chunk_id']
            } for chunk in host_content],
            ids=[chunk['chunk_id'] for chunk in host_content]
        )
    
    def retrieve_relevant_context(self, query: str, n_results: int = 5) -> List[str]:
        """Retrieve relevant context for poll generation"""
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        return results['documents'][0] if results['documents'] else []

class PollGenerator:
    def __init__(self, gemini_api_key: str):
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    def generate_contextual_polls(self, context_chunks: List[str], num_questions: int = 5) -> List[Dict]:
        """Generate high-quality polls based on host context using RAG"""
        if not context_chunks:
            return []
        
        # Combine context chunks
        combined_context = "\n\n".join(context_chunks)
        
        prompt = f"""
        Based STRICTLY on the following educational content spoken by the instructor, generate {num_questions} high-quality, thought-provoking multiple-choice questions. 

        CONTEXT (Instructor's content):
        {combined_context}

        REQUIREMENTS:
        1. Questions must be directly based ONLY on the provided context
        2. Create challenging questions that test deep understanding, not just recall
        3. Include questions about concepts, applications, comparisons, and implications
        4. Each question should have 4 options with only one correct answer
        5. Avoid generic or trivial questions
        6. Focus on the key learning objectives from the instructor's explanations

        FORMAT your response as a JSON array:
        [
            {{
                "question": "Your challenging question here",
                "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
                "correct_answer": "A",
                "explanation": "Brief explanation of why this is correct",
                "difficulty": "medium/hard",
                "concept": "Main concept being tested"
            }}
        ]

        Generate questions now:
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Extract JSON from response
            response_text = response.text
            
            # Find JSON array in response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                questions = json.loads(json_str)
                
                # Add unique IDs to questions
                for i, q in enumerate(questions):
                    q['id'] = str(uuid.uuid4())
                    q['created_at'] = datetime.now().isoformat()
                
                return questions
            else:
                st.error("Could not parse questions from AI response")
                return []
                
        except Exception as e:
            st.error(f"Error generating polls: {str(e)}")
            return []

class EngagementAnalyzer:
    def __init__(self):
        pass
    
    def calculate_speaking_time(self, transcripts: List[Dict]) -> Dict[str, Dict]:
        """Calculate speaking statistics for each participant"""
        speaker_stats = defaultdict(lambda: {
            'word_count': 0,
            'speaking_instances': 0,
            'total_chars': 0,
            'avg_response_length': 0
        })
        
        for transcript in transcripts:
            speaker = transcript['speaker']
            text = transcript['text']
            
            speaker_stats[speaker]['word_count'] += len(text.split())
            speaker_stats[speaker]['speaking_instances'] += 1
            speaker_stats[speaker]['total_chars'] += len(text)
        
        # Calculate percentages and averages
        total_words = sum(stats['word_count'] for stats in speaker_stats.values())
        
        for speaker, stats in speaker_stats.items():
            stats['percentage'] = (stats['word_count'] / total_words * 100) if total_words > 0 else 0
            stats['avg_response_length'] = stats['total_chars'] / stats['speaking_instances'] if stats['speaking_instances'] > 0 else 0
        
        return dict(speaker_stats)
    
    def create_engagement_pie_chart(self, speaker_stats: Dict) -> go.Figure:
        """Create pie chart for student engagement"""
        # Filter out host from student engagement
        students = {k: v for k, v in speaker_stats.items() if k != st.session_state.get('host_name', '')}
        
        if not students:
            return go.Figure()
        
        speakers = list(students.keys())
        percentages = [students[speaker]['percentage'] for speaker in speakers]
        
        fig = go.Figure(data=[go.Pie(
            labels=speakers,
            values=percentages,
            hole=0.3,
            textinfo='label+percent',
            textfont_size=12,
            marker=dict(
                colors=px.colors.qualitative.Set3,
                line=dict(color='#000000', width=2)
            )
        )])
        
        fig.update_layout(
            title={
                'text': 'Student Engagement Distribution',
                'x': 0.5,
                'xanchor': 'center',
                'font_size': 20
            },
            showlegend=True,
            height=500
        )
        
        return fig
    
    def create_ranking_table(self, speaker_stats: Dict) -> pd.DataFrame:
        """Create ranking table for students"""
        # Filter out host
        students = {k: v for k, v in speaker_stats.items() if k != st.session_state.get('host_name', '')}
        
        ranking_data = []
        for speaker, stats in students.items():
            ranking_data.append({
                'Rank': 0,  # Will be set after sorting
                'Student': speaker,
                'Participation %': round(stats['percentage'], 2),
                'Word Count': stats['word_count'],
                'Speaking Instances': stats['speaking_instances'],
                'Avg Response Length': round(stats['avg_response_length'], 1)
            })
        
        # Sort by participation percentage
        ranking_df = pd.DataFrame(ranking_data)
        ranking_df = ranking_df.sort_values('Participation %', ascending=False)
        ranking_df['Rank'] = range(1, len(ranking_df) + 1)
        
        return ranking_df[['Rank', 'Student', 'Participation %', 'Word Count', 'Speaking Instances', 'Avg Response Length']]

def load_sample_data():
    """Load the sample dataset"""
    try:
        # Read the uploaded file
        data = json.loads(open('sample_dataset.json', 'r').read())
        return data
    except:
        # Fallback sample data structure
        return {
            "session_id": "sample-session",
            "transcripts": [],
            "participants": [],
            "host": "",
            "total_duration_minutes": 0,
            "topic": "Sample Session"
        }

def main():
    st.title("Automatic Polling System with Student Engagement Analytics")
    st.markdown("---")
    
    # Initialize session state
    if 'transcript_data' not in st.session_state:
        st.session_state.transcript_data = None
    if 'polls_generated' not in st.session_state:
        st.session_state.polls_generated = []
    if 'poll_responses' not in st.session_state:
        st.session_state.poll_responses = {}
    if 'auto_generate_polls' not in st.session_state:
        st.session_state.auto_generate_polls = False
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Team Imagineers")
        
        # Get Gemini API Key from environment or input
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        # if not gemini_api_key:
        #     gemini_api_key = st.text_input(
        #         "Gemini API Key",
        #         type="password",
        #         help="Enter your Google Gemini API key or set GEMINI_API_KEY in .env file"
        #     )
        # else:
        #     st.success("✅ API Key loaded from environment")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Transcript JSON",
            type=['json'],
            help="Upload a JSON file containing session transcripts"
        )
        
        # Load sample data button
        if st.button("Use Sample Data", type="secondary"):
            try:
                # Use the provided sample data
                sample_data = {
                    "session_id": "0cbe1052-8c02-4602-895e-68e1b1a3b031",
                    "transcripts": [
                        {
                            "speaker": "Keerthana",
                            "text": "Good morning everyone! Today we're going to dive deep into machine learning fundamentals. Let's start with understanding what makes a good machine learning model. The key principle is the bias-variance tradeoff. High bias models are too simplistic and underfit the data, while high variance models are too complex and overfit. The sweet spot is finding the right balance that minimizes both bias and variance to achieve optimal generalization performance.",
                            "timestamp": "2024-01-20T10:00:00",
                            "session_id": "0cbe1052-8c02-4602-895e-68e1b1a3b031"
                        },
                        {
                            "speaker": "Rafi",
                            "text": "Professor, can you explain more about how we identify if our model is overfitting?",
                            "timestamp": "2024-01-20T10:00:45",
                            "session_id": "0cbe1052-8c02-4602-895e-68e1b1a3b031"
                        },
                        {
                            "speaker": "Keerthana",
                            "text": "Excellent question, Rafi! Overfitting typically manifests when your training accuracy is very high, but validation accuracy is significantly lower. We use techniques like cross-validation to detect this. Another telltale sign is when your learning curves show the training error continuing to decrease while validation error starts increasing. This gap indicates your model is memorizing training data rather than learning generalizable patterns. Regularization techniques like L1, L2, dropout, and early stopping help combat overfitting.",
                            "timestamp": "2024-01-20T10:00:53",
                            "session_id": "0cbe1052-8c02-4602-895e-68e1b1a3b031"
                        },
                        {
                            "speaker": "Sreejith",
                            "text": "What's the difference between L1 and L2 regularization in practical terms?",
                            "timestamp": "2024-01-20T10:01:48",
                            "session_id": "0cbe1052-8c02-4602-895e-68e1b1a3b031"
                        },
                        {
                            "speaker": "Keerthana",
                            "text": "Great follow-up question! L1 regularization, also called Lasso, adds the sum of absolute values of parameters to the loss function. This creates sparse models by driving some weights to exactly zero, effectively performing feature selection. L2 regularization, or Ridge regression, adds the sum of squared parameters. It shrinks weights towards zero but rarely makes them exactly zero. L1 is better when you want automatic feature selection and interpretable models with fewer features. L2 works well when all features are somewhat relevant and you want to prevent any single feature from dominating.",
                            "timestamp": "2024-01-20T10:02:00",
                            "session_id": "0cbe1052-8c02-4602-895e-68e1b1a3b031"
                        }
                    ],
                    "participants": ["Keerthana", "Rafi", "Sreejith"],
                    "host": "Keerthana",
                    "total_duration_minutes": 21,
                    "topic": "Machine Learning Fundamentals - Bias-Variance Tradeoff and Regularization"
                }
                st.session_state.transcript_data = sample_data
                st.session_state.host_name = sample_data['host']
                st.success("Sample data loaded successfully!")
                
                # Auto-generate polls if API key is available
                if gemini_api_key:
                    st.session_state.auto_generate_polls = True
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")
    
    # Load data from uploaded file
    if uploaded_file is not None:
        try:
            transcript_data = json.load(uploaded_file)
            st.session_state.transcript_data = transcript_data
            st.session_state.host_name = transcript_data.get('host', '')
            st.success("Data loaded successfully!")
            
            # Auto-generate polls if API key is available
            if gemini_api_key:
                st.session_state.auto_generate_polls = True
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    # Main interface
    if st.session_state.transcript_data is None:
        st.info("👆 Please upload a transcript file or use sample data to get started.")
        return
    
    if not gemini_api_key:
        st.warning("⚠️ Please enter your Gemini API key in the sidebar to generate polls.")
        return
    
    data = st.session_state.transcript_data
    
    # Display session info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Session Topic", data.get('topic', 'N/A'))
    with col2:
        st.metric("Duration", f"{data.get('total_duration_minutes', 0)} min")
    with col3:
        st.metric("Participants", len(data.get('participants', [])))
    
    st.markdown("---")
    
    # Auto-generate polls if triggered
    if st.session_state.auto_generate_polls and gemini_api_key and st.session_state.transcript_data:
        with st.spinner("Auto-generating polls from transcript..."):
            try:
                # Initialize components
                chroma_client = init_chromadb()
                embedding_model = load_embedding_model()
                
                # Process transcript
                processor = TranscriptProcessor(chroma_client, embedding_model)
                processor.setup_collection()
                
                # Extract host content
                host_content = processor.chunk_host_content(
                    st.session_state.transcript_data['transcripts'], 
                    st.session_state.transcript_data['host']
                )
                
                if host_content:
                    # Store in ChromaDB
                    processor.store_context(host_content)
                    
                    # Retrieve relevant context
                    context_chunks = processor.retrieve_relevant_context(
                        "educational concepts and key learning points",
                        n_results=min(len(host_content), 8)
                    )
                    
                    # Generate polls
                    poll_generator = PollGenerator(gemini_api_key)
                    polls = poll_generator.generate_contextual_polls(
                        context_chunks, 
                        5  # Default 5 questions
                    )
                    
                    if polls:
                        st.session_state.polls_generated = polls
                        st.success(f"Auto-generated {len(polls)} contextual polls!")
                    else:
                        st.warning("Could not generate polls from the transcript.")
                else:
                    st.warning("No host content found in the transcript.")
                    
            except Exception as e:
                st.error(f"Error auto-generating polls: {str(e)}")
            finally:
                st.session_state.auto_generate_polls = False
    tab1, tab2, tab3 = st.tabs(["📝 Poll Generation", "📊 Student Dashboard", "📈 Host Analytics"])
    
    with tab1:
        st.header("Generate Contextual Polls")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            num_questions = st.slider("Number of Questions", 1, 10, 5)
        with col2:
            if st.button("🎯 Generate Polls", type="primary"):
                with st.spinner("Processing transcript and generating polls..."):
                    try:
                        # Initialize components
                        chroma_client = init_chromadb()
                        embedding_model = load_embedding_model()
                        
                        # Process transcript
                        processor = TranscriptProcessor(chroma_client, embedding_model)
                        processor.setup_collection()
                        
                        # Extract host content
                        host_content = processor.chunk_host_content(
                            data['transcripts'], 
                            data['host']
                        )
                        
                        if not host_content:
                            st.error("No host content found to generate polls from.")
                            return
                        
                        # Store in ChromaDB
                        processor.store_context(host_content)
                        
                        # Retrieve relevant context
                        context_chunks = processor.retrieve_relevant_context(
                            "machine learning concepts and techniques",
                            n_results=min(len(host_content), 8)
                        )
                        
                        # Generate polls
                        poll_generator = PollGenerator(gemini_api_key)
                        polls = poll_generator.generate_contextual_polls(
                            context_chunks, 
                            num_questions
                        )
                        
                        if polls:
                            st.session_state.polls_generated = polls
                            st.success(f"Generated {len(polls)} contextual polls!")
                        else:
                            st.error("Failed to generate polls. Please check your API key and try again.")
                            
                    except Exception as e:
                        st.error(f"Error generating polls: {str(e)}")
        
        # Display generated polls
        if st.session_state.polls_generated:
            st.subheader("Generated Polls")
            
            for i, poll in enumerate(st.session_state.polls_generated):
                st.markdown(f"### Question {i+1}: {poll['concept']}")
                st.write(f"**{poll['question']}**")
                st.write(f"*Difficulty: {poll.get('difficulty', 'medium')}*")
                
                # Display options
                for option in poll['options']:
                    st.write(f"• {option}")
                
                # Show answer and explanation in a separate container
                st.markdown("**Answer & Explanation:**")
                st.info(f"**Correct Answer:** {poll['correct_answer']}")
                st.info(f"**Explanation:** {poll.get('explanation', 'No explanation provided')}")
                st.markdown("---")
    
    with tab2:
        st.header("Student Poll Interface")
        
        if not st.session_state.polls_generated:
            st.info("No polls available. Generate polls first in the Poll Generation tab.")
        else:
            st.write("**Instructions:** Answer the following questions based on today's session content.")
            
            # Student name input
            student_name = st.text_input("Enter your name:", key="student_name")
            
            if student_name:
                # Display polls for students
                student_responses = {}
                
                for i, poll in enumerate(st.session_state.polls_generated):
                    st.subheader(f"Question {i+1}")
                    st.write(poll['question'])
                    
                    # Radio buttons for options
                    selected_option = st.radio(
                        "Select your answer:",
                        poll['options'],
                        key=f"poll_{i}_{student_name}",
                        index=None
                    )
                    
                    if selected_option:
                        student_responses[f"question_{i}"] = {
                            'answer': selected_option,
                            'correct': selected_option.startswith(poll['correct_answer'])
                        }
                
                # Submit responses
                if st.button("Submit Responses", type="primary") and student_responses:
                    if student_name not in st.session_state.poll_responses:
                        st.session_state.poll_responses[student_name] = {}
                    
                    st.session_state.poll_responses[student_name] = student_responses
                    
                    # Calculate score
                    correct_answers = sum(1 for resp in student_responses.values() if resp['correct'])
                    total_questions = len(student_responses)
                    score = (correct_answers / total_questions) * 100
                    
                    st.success(f"Responses submitted! Score: {score:.1f}% ({correct_answers}/{total_questions})")
    
    with tab3:
        st.header("Host Analytics Dashboard")
        
        # Initialize engagement analyzer
        analyzer = EngagementAnalyzer()
        
        # Calculate engagement statistics
        speaker_stats = analyzer.calculate_speaking_time(data['transcripts'])
        
        # Create two columns for charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Student Engagement Distribution")
            pie_chart = analyzer.create_engagement_pie_chart(speaker_stats)
            if pie_chart.data:
                st.plotly_chart(pie_chart, use_container_width=True)
            else:
                st.info("No student data available for pie chart.")
        
        with col2:
            st.subheader("Engagement Metrics")
            # Overall stats
            total_participants = len([s for s in speaker_stats.keys() if s != data['host']])
            avg_participation = np.mean([stats['percentage'] for speaker, stats in speaker_stats.items() if speaker != data['host']]) if total_participants > 0 else 0
            
            st.metric("Total Students", total_participants)
            st.metric("Avg Participation", f"{avg_participation:.1f}%")
            st.metric("Most Active Student", 
                     max([s for s in speaker_stats.keys() if s != data['host']], 
                         key=lambda x: speaker_stats[x]['percentage']) if total_participants > 0 else "N/A")
        
        # Ranking table
        st.subheader("Student Ranking")
        ranking_df = analyzer.create_ranking_table(speaker_stats)
        if not ranking_df.empty:
            st.dataframe(
                ranking_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Add download button for rankings
            csv = ranking_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Rankings as CSV",
                data=csv,
                file_name=f"student_rankings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No student ranking data available.")
        
        # Poll Results (if available)
        if st.session_state.poll_responses:
            st.subheader("Poll Results")
            
            # Create results summary
            results_data = []
            for student, responses in st.session_state.poll_responses.items():
                correct_count = sum(1 for resp in responses.values() if resp['correct'])
                total_count = len(responses)
                score = (correct_count / total_count) * 100 if total_count > 0 else 0
                
                results_data.append({
                    'Student': student,
                    'Score (%)': round(score, 1),
                    'Correct Answers': correct_count,
                    'Total Questions': total_count
                })
            
            results_df = pd.DataFrame(results_data)
            results_df = results_df.sort_values('Score (%)', ascending=False)
            
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            # Poll performance chart
            if len(results_df) > 0:
                fig = px.bar(
                    results_df, 
                    x='Student', 
                    y='Score (%)',
                    title='Student Poll Performance',
                    color='Score (%)',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()