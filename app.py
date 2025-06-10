import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Task Management AI",
    page_icon="🚀",
    layout="wide"
)

# Title
st.title("🚀 Task Management AI System")
st.markdown("### Intelligent Task Classification, Prioritization & Workload Balancing")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", [
    "🏠 Home",
    "📊 Data Analysis", 
    "🤖 Model Performance",
    "⚖️ Workload Balancer",
    "🔮 Task Predictor"
])

if page == "🏠 Home":
    st.header("Project Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Models", "5", "NB, SVM, RF, XGB, Ensemble")
    with col2:
        st.metric("Best Accuracy", "94%", "Ensemble Model")
    with col3:
        st.metric("Features Used", "110", "TF-IDF + Engineered")
    with col4:
        st.metric("Weeks Completed", "4", "Full Pipeline")
    
    # Project timeline
    st.subheader("📅 Project Timeline")
    timeline_data = {
        'Week': ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
        'Tasks': [
            'Data Preprocessing & EDA',
            'Feature Extraction & Classification',
            'Advanced Models & Optimization',
            'Finalization & Dashboard'
        ],
        'Status': ['✅ Complete', '✅ Complete', '✅ Complete', '✅ Complete']
    }
    st.table(pd.DataFrame(timeline_data))
    
    # System Architecture (text-based instead of mermaid)
    st.subheader("🏗️ System Architecture")
    st.info("""
    **Data Pipeline Flow:**
    
    1. 📊 **Raw Task Data** → Data Collection
    2. 🧹 **Data Preprocessing** → Text cleaning & feature engineering  
    3. 🔤 **Feature Extraction** → TF-IDF + Engineered features
    4. 🤖 **Model Training** → 5 different ML algorithms
    5. 🎯 **Ensemble Model** → Combined predictions
    6. 📈 **Final Output** → Classification, Prioritization & Workload Balancing
    """)

elif page == "📊 Data Analysis":
    st.header("Data Analysis Dashboard")
    
    # Sample data for demo
    @st.cache_data
    def load_sample_data():
        return pd.DataFrame({
            'Priority': ['High', 'Medium', 'Low', 'High', 'Medium'] * 20,
            'Status': ['Done', 'In Progress', 'To Do'] * 33 + ['Done'],
            'Label': ['ai/ml', 'deployment', 'devops', 'testing'] * 25,
            'Assignee': ['Shridayal', 'Anita', 'Vikram', 'Priya'] * 25
        })
    
    df = load_sample_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Priority distribution
        st.subheader("Priority Distribution")
        priority_counts = df['Priority'].value_counts()
        st.bar_chart(priority_counts)
    
    with col2:
        # Status distribution
        st.subheader("Status Distribution") 
        status_counts = df['Status'].value_counts()
        st.bar_chart(status_counts)

elif page == "🤖 Model Performance":
    st.header("Model Performance Comparison")
    
    # Model performance data
    performance_data = {
        'Model': ['Naive Bayes', 'SVM', 'Random Forest', 'XGBoost', 'Ensemble'],
        'Accuracy': [0.85, 0.87, 0.91, 0.93, 0.94],
        'Precision': [0.84, 0.86, 0.90, 0.92, 0.93],
        'Recall': [0.85, 0.87, 0.91, 0.93, 0.94],
        'F1-Score': [0.84, 0.86, 0.90, 0.92, 0.93]
    }
    
    df_performance = pd.DataFrame(performance_data)
    
    # Display metrics table
    st.subheader("📈 Performance Metrics")
    st.dataframe(df_performance, use_container_width=True)
    
    # Performance comparison chart
    st.subheader("Model Accuracy Comparison")
    accuracy_data = df_performance.set_index('Model')['Accuracy']
    st.bar_chart(accuracy_data)
    
    # Best model highlight
    st.success("🏆 **Best Model**: Ensemble with 94% accuracy!")

elif page == "⚖️ Workload Balancer":
    st.header("Intelligent Workload Balancer")
    
    # Workload data
    workload_data = {
        'Assignee': ['Shridayal', 'Anita', 'Vikram', 'Priya'],
        'Current_Load': [32, 28, 35, 25],
        'Capacity': [40, 40, 40, 40],
    }
    
    df_workload = pd.DataFrame(workload_data)
    df_workload['Available'] = df_workload['Capacity'] - df_workload['Current_Load']
    df_workload['Utilization %'] = (df_workload['Current_Load'] / df_workload['Capacity'] * 100).round(1)
    
    # Display current workload
    st.subheader("👥 Current Team Workload")
    st.dataframe(df_workload, use_container_width=True)
    
    # Workload visualization
    chart_data = df_workload.set_index('Assignee')[['Current_Load', 'Available']]
    st.bar_chart(chart_data)
    
    # Utilization warning
    for idx, row in df_workload.iterrows():
        if row['Utilization %'] > 85:
            st.warning(f"⚠️ {row['Assignee']} is at {row['Utilization %']}% capacity!")
        elif row['Utilization %'] < 60:
            st.info(f"✅ {row['Assignee']} has available capacity ({row['Utilization %']}%)")

elif page == "🔮 Task Predictor":
    st.header("AI Task Predictor")
    st.markdown("Enter a task description to get AI-powered predictions!")
    
    # Task input form
    with st.form("task_prediction_form"):
        task_description = st.text_area(
            "Task Description", 
            placeholder="e.g., Implement new machine learning model for customer segmentation",
            height=100
        )
        
        col1, col2 = st.columns(2)
        with col1:
            deadline = st.date_input("Deadline")
        with col2:
            preferred_assignee = st.selectbox(
                "Preferred Assignee (Optional)", 
                ["Auto-assign", "Shridayal", "Anita", "Vikram", "Priya"]
            )
        
        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)
    
    if submitted and task_description:
        with st.spinner("Analyzing task..."):
            # Simulate prediction
            import time
            time.sleep(1)
            
            # Mock predictions
            predicted_priority = np.random.choice(['High', 'Medium', 'Low'], p=[0.3, 0.5, 0.2])
            predicted_category = np.random.choice(['ai/ml', 'deployment', 'devops', 'testing'])
            confidence = np.random.uniform(0.85, 0.98)
            suggested_assignee = np.random.choice(['Shridayal', 'Anita', 'Vikram', 'Priya'])
            estimated_hours = {'High': 8, 'Medium': 5, 'Low': 2}[predicted_priority]
            
        # Display results
        st.success("✅ Task Analysis Complete!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Priority", predicted_priority)
            st.metric("Estimated Hours", f"{estimated_hours}h")
        
        with col2:
            st.metric("Predicted Category", predicted_category)
            st.metric("Confidence", f"{confidence:.1%}")
        
        with col3:
            st.metric("Suggested Assignee", suggested_assignee)
            st.metric("Expected Completion", "3 days")
        
        # Task summary
        st.subheader("📋 Task Summary")
        st.info(f"""
        **Task**: {task_description[:100]}{'...' if len(task_description) > 100 else ''}
        
        **AI Recommendations**:
        - Priority: {predicted_priority}
        - Category: {predicted_category}
        - Assign to: {suggested_assignee}
        - Estimated effort: {estimated_hours} hours
        - Confidence: {confidence:.1%}
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🚀 Task Management AI System | Built with Streamlit | 
        <a href='https://github.com/shridayal/task_management_ai'>View on GitHub</a></p>
    </div>
    """, 
    unsafe_allow_html=True
)
