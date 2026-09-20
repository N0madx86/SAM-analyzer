const analyzeBtn = document.getElementById("analyzeBtn");

const reviewInput = document.getElementById("review");

const loading = document.getElementById("loading");
const result = document.getElementById("result");
const error = document.getElementById("error");

const overallSentiment =
    document.getElementById("overallSentiment");

const positiveRatio =
    document.getElementById("positiveRatio");

const negativeRatio =
    document.getElementById("negativeRatio");

const models =
    document.getElementById("models");


analyzeBtn.addEventListener("click", analyze);


async function analyze() {

    const text = reviewInput.value.trim();

    if (!text) {

        error.textContent =
            "Please enter a review.";

        result.hidden = true;

        return;
    }

    error.textContent = "";

    loading.hidden = false;
    result.hidden = true;

    try {

        const response = await fetch(
            "http://127.0.0.1:8000/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    text: text
                })
            }
        );

        if (!response.ok) {
            throw new Error(
                `API error: ${response.status}`
            );
        }

        const data = await response.json();

        displayResult(data);

    } catch (err) {

        error.textContent =
            "Unable to connect to the sentiment API.";

        console.error(err);

    } finally {

        loading.hidden = true;
    }
}


function displayResult(data) {

    const overall = data.overall;

    overallSentiment.textContent =
        `Overall: ${overall.sentiment}`;

    positiveRatio.textContent =
        `${(overall.positive_ratio * 100).toFixed(1)}%`;

    negativeRatio.textContent =
        `${(overall.negative_ratio * 100).toFixed(1)}%`;

    models.innerHTML = "";

    for (
        const [name, prediction]
        of Object.entries(data.models)
    ) {

        const modelElement =
            document.createElement("p");

        modelElement.textContent =
            `${name}: ${prediction.sentiment} ` +
            `(${(prediction.confidence * 100).toFixed(1)}%)`;

        models.appendChild(modelElement);
    }

    result.hidden = false;
}