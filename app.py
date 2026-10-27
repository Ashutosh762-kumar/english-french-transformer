import streamlit as st
from translation import translate_sentence


st.set_page_config(
    page_title="Translator",
    page_icon="💬",
    layout="centered",

)

st.title("Translator")
st.subheader("English to French Translator")

st.write("Translate your text from English to French using a Transformer model.")

st.divider()


english_text = st.text_area(
            "Enter English Text",
            placeholder="Example : Hello, how are you?",
            height=150,
)

if st.button("Translate",use_container_width=True):

    if not english_text.strip():
        st.warning("Please enter an english sentence.")
    else:
        with st.spinner("Translating...."):
            try:
                st.session_state["translation"] = translate_sentence(english_text)
            except Exception as e:
                st.error(f"Translation failed: {e}")

st.divider()
st.subheader("French Translation")


if "translation" in st.session_state:
    st.text_area(
        "Result",
        value=st.session_state["translation"],
        height=150,
        disabled=True
    )

    st.button(
        "Copy translation",
        on_click=lambda: st.session_state.update(
            copied_translation=st.session_state["translation"]
        )
    )

    if "copied_translation" in st.session_state:
        st.caption("Select and copied the translation above")
else:
    st.caption("Your translation will appear here")

