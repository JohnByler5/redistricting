document.addEventListener("DOMContentLoaded", function() {
    document.querySelectorAll(".parameter-container input[type='range']").forEach(slider => {
        const display = slider.nextElementSibling;
        const isPercentage = display.textContent.includes("%");
        slider.addEventListener('input', () => updateSliderValue(slider, display, isPercentage));
        updateSliderValue(slider, display, isPercentage);
    });
});

function updateSliderValue(slider, display, isPercentage = false) {
    display.textContent = isPercentage ? `${slider.value}%` : parseInt(slider.value).toLocaleString();
}

function resetParameters() {
    const elementsToReset = [
        { id: "state-select", value: "PA" },
        { id: "generation-slider", value: 100000, displayId: "generation-value" },
        { id: "population-slider", value: 2, displayId: "population-value" },
        { id: "starting-population-slider", value: 25, displayId: "starting-population-value" },
        { id: "population-selection-slider", value: 50, displayId: "population-selection-value", isPercentage: true },
        { id: "contiguity-slider", value: 0, displayId: "contiguity-value" },
        { id: "pop-balance-slider", value: -5, displayId: "pop-balance-value" },
        { id: "compactness-slider", value: 1, displayId: "compactness-value" },
        { id: "win-margin-slider", value: -1, displayId: "win-margin-value" },
        { id: "efficiency-gap-slider", value: -1, displayId: "efficiency-gap-value" },
    ];

    elementsToReset.forEach(({ id, value, displayId, isPercentage }) => {
        const element = document.getElementById(id);
        element.value = value;
        if (displayId) {
            const displayElement = document.getElementById(displayId);
            updateSliderValue(element, displayElement, isPercentage);
        }
    });
}

function startAlgorithm() {
    document.getElementById("start-running").style.display = "none";
    document.getElementById("stop-running").style.display = "inline";
    window.scrollTo({
      top: document.getElementById("time").offsetTop - 50,
      left: 0,
      behavior: 'smooth' // Optional: Adds a smooth scrolling effect
    });

    fetch('/start-algorithm', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => console.log(data.message))
    .catch((error) => {
        console.error('Error:', error);
    });
}

function stopAlgorithm() {
    document.getElementById("start-running").style.display = "inline";
    document.getElementById("stop-running").style.display = "none";

    fetch('/stop-algorithm', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => console.log(data.message))
    .catch((error) => {
        console.error('Error:', error);
    });
}

function updateMapAndStats(mapType, mapData) {
    const mapImage = document.querySelector(`.${mapType} img`);
    mapImage.src = mapData.imageUrl;
    const statsContainer = document.querySelector(`.${mapType} .stats-container`);
    for (const statName in mapData.stats) {
        const statValue = mapData.stats[statName];
        const statElement = statsContainer.querySelector(`.${statName} .value`);
        statElement.textContent = statValue;
    }
}

function connectEventSource() {
    const headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',
    }
    const eventSource = new EventSource("/events", {headers: headers, heartbeatTimeout: 1000 * 60 * 2});

    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        if ("event" in data && ["USER_ID_NOT_FOUND", "OPERATION_COMPLETE"].includes(data["event"])) {
            document.getElementById("start-running").style.display = "inline";
            document.getElementById("stop-running").style.display = "none";
        } else if ("event" in data) {
            console.log(data);
        } else {
            document.getElementById('time').textContent = data.timeElapsed;
            document.getElementById('generation').textContent = data.generation;
            updateMapAndStats('current-map', data.currentMap);
            updateMapAndStats('solution-map', data.solutionMap);
        }
    };

    eventSource.onerror = function(error) {
        console.error("EventSource failed:", error);
        eventSource.close();
        connectEventSource();
    };
}

connectEventSource();
