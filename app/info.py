about = [
    "This app is part of the [H3D Symposium (Livingstone, Zambia, 2024)](https://h3dfoundation.org/5th-h3d-symposium/).",
    "The workshop been jointly developed by the [Ersilia Open Source Initiative](https://ersilia.io) and the [H3D Foundation](https://h3dfoundation.org/).",
    "For more information about this workshop, please see code and data in this [GitHub repository](https://github.com/ersilia-os/ai-intro-workshop).",
    "If you have a more advanced dataset in mind or a use case for your research, please contact us at: [hello@ersilia.io](mailto:hello@ersilia.io)."
]

model_urls = {
    "eos9ei3": ["https://eos9ei3-jvhi9.ondigitalocean.app/",
                "https://eos9ei3-2-yml5d.ondigitalocean.app/",
                "https://eos9ei3-3-figt3.ondigitalocean.app/"
                ],
                
    "eos43at": ["https://eos43at-boaoi.ondigitalocean.app/",
                "https://oyster-app-p5b7w.ondigitalocean.app/",
                "https://eos43at-3-tleva.ondigitalocean.app/"
                ]
}


library_filenames = {
    "Compound library 1": "abaumannii/abaumannii_subset250_0.csv",
    "Compound library 2": "abaumannii/abaumannii_subset250_1.csv",
    "Compound library 3": "abaumannii/abaumannii_subset250_2.csv",
    "Compound library 4": "abaumannii/abaumannii_subset250_3.csv",
}

q1 = [
    "- What is the experimental assay measuring?",
    "- Are we training a classification or regression model with this data?",
    "- What is the author's defined activity cut-off?",
    "- What have we chosen as a cut-off for activity against A. baumannii?",
    "- Is it a balanced dataset? Why or why not?",
    "- What is the performance of the models at different cut-offs?",
    "- How does our quick modelling compare to the author's work?"
]

q2 = [
    "- What type of model is it?",
    "- What type of output does the model produce?",
    "- What is a good threshold for keeping molecules?"
]

q3 = [
    "- Are there new chemical structures in the generated candidates?",
    "- Do the generated candidates present better predicted biochemical profiles?",
    "- What would your next steps be?"
]

# Markdown content for A. baumannii Bioactivity section
abaumannii_bioactivity = """
<div style="background-color: lightblue; padding: 10px; border-radius: 5px;">
    <h5>🦠 A. baumannii Bioactivity</h5>
    <p>This is the model we just trained on a dataset of ~7500 molecules described in Liu et al, 2023 to elucidate whether a molecule is active against A.baumannii.</p>
</div>
"""

# Markdown content for hERG Inhibition section
herg_inhibition = """
<div style="background-color: lightblue; padding: 10px; border-radius: 5px;">
    <h5>🫀 hERG Inhibition</h5>
    <p>This model is described in Jiménez-Luna et al, 2021. It was trained on a publicly available dataset with the goal of predicting hERG-mediated cardiotoxicity.</p>
</div>
"""

# Markdown content for Synthetic Accessibility section
synthetic_accessibility = """
<div style="background-color: lightblue; padding: 10px; border-radius: 5px;">
    <h5>🧪 Synthetic Accessibility</h5>
    <p>The synthetic accessibility score was developed by Ertl & Schuffenhauer, 2009. It estimates if a molecule will be accessible for synthesis in the laboratory.</p>
</div>
"""
