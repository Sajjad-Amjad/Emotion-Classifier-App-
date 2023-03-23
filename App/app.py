# Core Pkgs
import streamlit as st
import altair as alt

# NLP pkgs
import numpy as np
import pandas as pd

# Utils
import joblib

pipe_lr = joblib.load(open('models/emotion_pipe_lr.pkl', 'rb'))
emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}

def predict_emotion(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def prediction_probability(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def main():
    menu = ["Home", "Moniter", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader('Home-Emotion Text Classifier')
        with st.form(key="emotion_clf_form"):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label="Submit")
        if submit_text:
            col1, col2 = st.columns(2)

            # Applyinh functions here
            prediction = predict_emotion(raw_text)
            probability = prediction_probability(raw_text)


            with col1:
                st.success("Original")
                st.write(raw_text)
                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction, emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))

            with col2:
                st.success("Prediction Probability")
                # st.write(probability)
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                # st.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)




    elif choice == "Moniter":
        st.subheader("Moniter")

    else:
        st.subheader("About")




if __name__ == "__main__":
    main()