import time
import streamlit as st

from pipeline.retrieval import ask_all_models


##########################################################
# Page Configuration
##########################################################

st.set_page_config(

    page_title="Pharmacovigilance GenAI Assistant",

    page_icon="💊",

    layout="wide",

    initial_sidebar_state="expanded",

)

##########################################################
# Custom CSS
##########################################################

st.markdown(
    """
<style>

.block-container{
    padding-top:2rem;
}

.metric-box{
    border:1px solid #cccccc;
    border-radius:10px;
    padding:10px;
    background-color:#f8f9fa;
}

.answer-box{
    border-radius:8px;
    padding:15px;
    border:1px solid #dddddd;
}

</style>
""",
    unsafe_allow_html=True,
)

##########################################################
# Session State
##########################################################

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []

##########################################################
# Sidebar
##########################################################

with st.sidebar:

    st.title("💊 PV Assistant")

    st.markdown("---")

    st.subheader("Project")

    st.success("Vector Database Loaded")

    st.success("Embeddings Loaded")

    st.success("Multi-LLM Enabled")

    st.markdown("---")

    st.subheader("Embedding Model")

    st.info("sentence-transformers/all-MiniLM-L6-v2")

    st.markdown("---")

    st.subheader("Models")

    st.write("✅ GPT-4.1-mini")

    st.write("✅ GPT-4.1-nano")

    st.write("✅ Llama 3.2")

    st.write("✅ Qwen 2.5")

    st.markdown("---")

    st.subheader("Retriever")

    st.write("ChromaDB")

    st.write("Top K = 5")

    st.markdown("---")

    if st.button("🗑 Clear Chat"):

        st.session_state.chat_history = []

        st.rerun()

##########################################################
# Main Page
##########################################################

st.title("💊 Pharmacovigilance GenAI Assistant")

st.caption(
    "Retrieval-Augmented Generation (RAG) for Pharmacovigilance Case Summarization"
)

st.markdown("---")

##########################################################
# User Input
##########################################################

question = st.text_area(

    "Ask a Question",

    placeholder="""
Examples

• Summarize case CASE001

• Show the adverse events for CASE045

• Explain the causality assessment.

• Retrieve similar cases involving Warfarin.

• What is the regulatory deadline?
""",

    height=180,

)

##########################################################
# Generate Button
##########################################################

generate = st.button(

    "Generate Summary",

    type="primary",

    use_container_width=True,

)

##########################################################
# Run RAG Pipeline
##########################################################

if generate:

    if question.strip() == "":

        st.warning("Please enter a question.")

        st.stop()

    with st.spinner("Retrieving documents and generating responses..."):

        start = time.time()

        result = ask_all_models(question)

        end = time.time()

        elapsed = round(end - start, 2)

        st.session_state.chat_history.append(

            {

                "question": question,

                "result": result,

                "time": elapsed,

            }

        )



##########################################################
# Display Latest Result
##########################################################

if len(st.session_state.chat_history) > 0:

    latest_chat = st.session_state.chat_history[-1]

    result = latest_chat["result"]

    elapsed = latest_chat["time"]

    st.markdown("---")

    ######################################################
    # Metrics
    ######################################################

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "LLMs Compared",
            len(result["answers"])
        )

    with col2:
        st.metric(
            "Retrieved Documents",
            len(result["documents"])
        )

    with col3:
        st.metric(
            "Response Time",
            f"{elapsed} sec"
        )

    st.markdown("---")

    ######################################################
    # LLM Responses
    ######################################################

    st.header("🤖 LLM Responses")

    model_names = list(result["answers"].keys())

    st.info(
    "All models received the same retrieved context from ChromaDB. "
    "The comparison below highlights differences in generation quality."
)

    tabs = st.tabs(model_names)

    for tab, model in zip(tabs, model_names):

        with tab:

            st.subheader(model)

            # st.markdown(result["answers"][model])
            with st.container(border=True):

                st.markdown(result["answers"][model])

            st.markdown("---")

            st.caption(
                f"Generated by {model}"
            )

    st.markdown("---")


    ##########################################################
    # Retrieved Evidence
    ##########################################################

    st.header("📄 Retrieved Evidence")

    st.info(
        "These are the documents retrieved from ChromaDB and supplied "
        "to every LLM. All models received exactly the same context."
    )

    for i, (doc, score) in enumerate(
        zip(result["documents"], result["scores"]),
        start=1,
    ):

        with st.expander(
            f"Document {i} • Similarity Score: {score:.4f}",
            expanded=False,
        ):

            col1, col2 = st.columns([1, 3])

            with col1:

                st.markdown("### Metadata")

                st.json(doc["metadata"])

            with col2:

                st.markdown("### Content")

                st.text(doc["content"])

    st.markdown("---")


    ##########################################################
    # Download Report
    ##########################################################

    st.header("📥 Download Report")

    report = f"""
    # Pharmacovigilance GenAI Assistant Report

    Question

    {latest_chat['question']}

    Response Time

    {elapsed} seconds

    ============================================================

    """

    for model, answer in result["answers"].items():

        report += f"""

    ============================================================
    {model}
    ============================================================

    {answer}

    """

    report += """

    ============================================================
    Retrieved Documents
    ============================================================

    """

    for i, (doc, score) in enumerate(
        zip(result["documents"], result["scores"]),
        start=1,
    ):

        report += f"""

    Document {i}

    Similarity Score:
    {score}

    Metadata:
    {doc["metadata"]}

    Content:
    {doc["content"]}

    ------------------------------------------------------------

    """

    st.download_button(

        label="⬇ Download Markdown Report",

        data=report,

        file_name="pharmacovigilance_summary.md",

        mime="text/markdown",

    )


    st.markdown("---")


    ##########################################################
    # Chat History
    ##########################################################

    st.header("💬 Chat History")

    for chat_number, chat in enumerate(
        reversed(st.session_state.chat_history),
        start=1,
    ):

        with st.expander(
            f"Question {len(st.session_state.chat_history)-chat_number+1}",
            expanded=False,
        ):

            st.markdown("### User Question")

            st.write(chat["question"])

            st.markdown("### Response Time")

            st.write(f"{chat['time']} seconds")

            st.markdown("### Models Used")

            st.write(", ".join(chat["result"]["answers"].keys()))

    st.markdown("---")


    ##########################################################
    # Footer
    ##########################################################

    st.caption(
        "Pharmacovigilance GenAI Assistant | "
        "Built using LangChain, ChromaDB, Hugging Face Embeddings, "
        "Multiple LLMs, and Streamlit"
    )