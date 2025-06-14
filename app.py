elif page == "Generate Handwriting":
        st.header("Generate Handwritten Text")
        
        model_files = []
        if os.path.exists('models'):
            model_files = [f for f in os.listdir('models') if f.endswith('.pth')]
        
        if not model_files:
            st.error("No trained models found! Please train a model first.")
            return
        
        selected_model = st.selectbox("Select a model", model_files)
        
        # Show current model status
        if st.session_state.model_loaded:
            st.success("✅ Model is loaded and ready!")
        else:
            st.info("⚠️ No model loaded")
        
        if st.button("Load Model"):
            with st.spinner("Loading model..."):
                if app.load_model(f'models/{selected_model}'):
                    st.success("Model loaded successfully!")
                    st.rerun()  # Refresh to show updated status
        
        text = st.text_input("Enter text to generate", "Hello World!")
        
        if st.button("Generate Handwriting"):
            if not st.session_state.model_loaded or st.session_state.model is None:
                st.error("Please load a model first!")
                return
                
            with st.spinner("Generating handwriting..."):
                try:
                    generated_image = app.generate_handwriting(text)
                    
                    if generated_image is not None:
                        fig, ax = plt.subplots(figsize=(12, 4))
                        ax.imshow(generated_image, cmap='gray')
                        ax.set_title(f'Generated: "{text}"', fontsize=14)
                        ax.axis('off')
                        st.pyplot(fig)
                        
                        # Convert to PIL Image for download
                        img_array = (generated_image * 255).astype(np.uint8)
                        img_pil = Image.fromarray(img_array)
                        
                        buf = io.BytesIO()
                        img_pil.save(buf, format='PNG')
                        byte_im = buf.getvalue()
                        
                        st.download_button(
                            "Download Generated Image",
                            data=byte_im,
                            file_name=f"handwriting_{text.replace(' ', '_')}.png",
                            mime="image/png"
                        )
                    else:
                        st.error("Failed to generate image")
                        
                except Exception as e:
                    st.error(f"Error generating handwriting: {str(e)}")
