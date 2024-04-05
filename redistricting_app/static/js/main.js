document.addEventListener("DOMContentLoaded", function() {
    sessionStorage.setItem('userID', crypto.randomUUID());

    document.querySelectorAll(".parameter-container input[type='range']").forEach(slider => {
        const display = slider.nextElementSibling;
        const reverse = ["contiguity-slider", "pop-balance-slider", "win-margin-slider", "efficiency-gap-slider"].includes(slider.id);
        const isPercentage = display.textContent.includes("%");
        slider.addEventListener('input', () => updateSliderValue(slider, display, reverse, isPercentage));
        updateSliderValue(slider, display, reverse, isPercentage);
    });
});

function updateSliderValue(slider, display, reverse = false, isPercentage = false) {
    let value = reverse ? -slider.value : slider.value;
    display.textContent = isPercentage ? `${slider.value}%` : parseInt(value).toLocaleString();
}

function resetParameters() {
    const elementsToReset = [
        { id: "state-select", value: "PA"},
        { id: "generation-slider", value: 10000, displayId: "generation-value", reverse: false},
        { id: "population-slider", value: 2, displayId: "population-value", reverse: false},
        { id: "starting-population-slider", value: 25, displayId: "starting-population-value", reverse: false},
        { id: "population-selection-slider", value: 50, displayId: "population-selection-value", reverse: false, isPercentage: true},
        { id: "contiguity-slider", value: 0, displayId: "contiguity-value", reverse: true},
        { id: "pop-balance-slider", value: 5, displayId: "pop-balance-value", reverse: true},
        { id: "compactness-slider", value: 1, displayId: "compactness-value", reverse: false},
        { id: "win-margin-slider", value: 1, displayId: "win-margin-value", reverse: true},
        { id: "efficiency-gap-slider", value: 1, displayId: "efficiency-gap-value", reverse: true},
    ];

    elementsToReset.forEach(({ id, value, displayId, reverse, isPercentage }) => {
        const element = document.getElementById(id);
        element.value = value;
        if (displayId) {
            const displayElement = document.getElementById(displayId);
            console.log(id, value, reverse, isPercentage)
            updateSliderValue(element, displayElement, reverse, isPercentage);
        }
    });
}

function startAlgorithm() {
    const userID = sessionStorage.getItem('userID');
    if (!userID) {
        console.error("User ID not found.");
        return;
    }

    document.getElementById("start-running").style.display = "none";
    document.getElementById("stop-running").style.display = "inline";
    window.scrollTo({
      top: document.getElementById("time").offsetTop - 50,
      left: 0,
      behavior: 'smooth'
    });

    const algorithmParameters = {
        "state": document.getElementById("state-select").value.toLowerCase(),
        "generations": parseInt(document.getElementById("generation-slider").value),
        "population_size": parseInt(document.getElementById("population-slider").value),
        "starting_population_size": parseInt(document.getElementById("starting-population-slider").value),
        "selection_pct": parseInt(document.getElementById("population-selection-slider").value) / 100,
        "weights": {
            "contiguity": -parseInt(document.getElementById("contiguity-slider").value),
            "population_balance": -parseInt(document.getElementById("pop-balance-slider").value),
            "compactness": parseInt(document.getElementById("compactness-slider").value),
            "win_margin": -parseInt(document.getElementById("win-margin-slider").value),
            "efficiency_gap": -parseInt(document.getElementById("efficiency-gap-slider").value),
        },
    };

    const params = {
        "user_id": userID,
        "algorithmParams": algorithmParameters,
    }

    fetch('/start-algorithm', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(params)
    })
    .then(response => response.json())
    .then(data => console.log(data.message))
    .catch((error) => {
        console.error('Error:', error);
    });
}


function stopAlgorithm() {
    const userID = sessionStorage.getItem('userID');
    if (!userID) {
        console.error("User ID not found.");
        return;
    }

    document.getElementById("stop-running").style.display = "none";
    document.getElementById("start-running").style.display = "inline";

    const params = {
        "user_id": userID,
    }

    fetch('/stop-algorithm', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(params)
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
    const userID = sessionStorage.getItem('userID');
    if (!userID) {
        console.error("User ID not found.");
        return;
    }

    const headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',
    }
    const eventSourceUrl = `/events?user_id=${encodeURIComponent(userID)}`;
    const eventSource = new EventSource(eventSourceUrl, {headers: headers, heartbeatTimeout: 1000 * 60 * 2});

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
