Const cvsIn = document.getElementById("inputimg");
const ctxIn = cvsIn.getContext("2d");
const divOut = document.getElementById("pred");

let svgGraph = null;
let isDrawing = false;
let currentMode = "one";

window.onload = () => {
  resetCanvas();
  initProbGraph();
};

function resetCanvas() {
  ctxIn.fillStyle = "white";
  ctxIn.fillRect(0, 0, cvsIn.width, cvsIn.height);

  ctxIn.strokeStyle = "black";
  ctxIn.lineWidth = 12;
  ctxIn.lineCap = "round";
  ctxIn.lineJoin = "round";
}

function setMode(mode) {
  currentMode = mode;

  if (mode === "one") {
    document.getElementById("modeText").innerText = "One Digit Detect";
    document.getElementById("finalResult").innerText = "Draw one digit";
  } else if (mode === "multi") {
    document.getElementById("modeText").innerText = "Multi Digit Detect";
    document.getElementById("finalResult").innerText = "Draw more than one digit";
  } else if (mode === "sum") {
    document.getElementById("modeText").innerText = "Sum Digits";
    document.getElementById("finalResult").innerText = "Draw digits to calculate their sum";
  } else if (mode === "word") {
    document.getElementById("modeText").innerText = "Detect Word";
    document.getElementById("finalResult").innerText = "Draw separated English letters";
  } else if (mode === "clearword") {
    onClear();
    currentMode = "one";
    document.getElementById("modeText").innerText = "One Digit Detect";
  }
}

function getPos(e) {
  const rect = cvsIn.getBoundingClientRect();

  if (e.touches && e.touches.length > 0) {
    return {
      x: e.touches[0].clientX - rect.left,
      y: e.touches[0].clientY - rect.top
    };
  }

  return {
    x: e.clientX - rect.left,
    y: e.clientY - rect.top
  };
}

function startDraw(e) {
  e.preventDefault();
  isDrawing = true;

  const pos = getPos(e);
  ctxIn.beginPath();
  ctxIn.moveTo(pos.x, pos.y);
}

function draw(e) {
  if (!isDrawing) return;

  e.preventDefault();
  const pos = getPos(e);

  ctxIn.lineTo(pos.x, pos.y);
  ctxIn.stroke();
}

function stopDraw(e) {
  if (!isDrawing) return;

  e.preventDefault();
  isDrawing = false;
  onRecognition();
}

cvsIn.addEventListener("mousedown", startDraw);
cvsIn.addEventListener("mousemove", draw);
cvsIn.addEventListener("mouseup", stopDraw);
cvsIn.addEventListener("mouseleave", stopDraw);

cvsIn.addEventListener("touchstart", startDraw);
cvsIn.addEventListener("touchmove", draw);
cvsIn.addEventListener("touchend", stopDraw);

cvsIn.addEventListener("contextmenu", e => e.preventDefault());

document.getElementById("clearbtn").onclick = onClear;

function onClear() {
  isDrawing = false;
  resetCanvas();

  divOut.textContent = "0";
  document.getElementById("prob").innerHTML = "Probability:";
  document.getElementById("finalResult").innerText = "Result will appear here";

  updateGraph([], null, false);
}

function onRecognition() {
  cvsIn.toBlob(async blob => {
    const body = new FormData();
    body.append("img", blob, "image.png");

    let url = "./DigitRecognition";

    if (currentMode === "one") {
      body.append("mode", "one");
    } else if (currentMode === "multi") {
      body.append("mode", "multi");
    } else if (currentMode === "sum") {
      body.append("mode", "sum");
    } else if (currentMode === "word") {
      url = "./LetterRecognition";
      body.append("mode", "word");
    }

    try {
      const response = await fetch(url, {
        method: "POST",
        body: body
      });

      const resjson = await response.json();
      showResult(resjson);

    } catch (error) {
      console.log(error);
      document.getElementById("finalResult").innerText = "Error: check server.py";
    }
  });
}

function showResult(res) {
  if (currentMode === "one") {
    divOut.textContent = res.pred;

    document.getElementById("prob").innerHTML =
      "Probability: " + res.probs[res.pred].toFixed(2) + "%";

    document.getElementById("finalResult").innerText =
      "Detected digit: " + res.pred;

    updateGraph(res.probs, res.pred, false);
  }

  else if (currentMode === "multi") {
    divOut.textContent = res.multi || "-";

    document.getElementById("prob").innerHTML = "Detected Number";

    document.getElementById("finalResult").innerText =
      "Detected number: " + (res.multi || "No digits detected");

    updateGraph(res.probs, null, false);
  }

  else if (currentMode === "sum") {
    divOut.textContent = res.sum;

    document.getElementById("prob").innerHTML = "Digit Sum";

    document.getElementById("finalResult").innerText =
      "Digits: " + (res.multi || "-") + " → Sum = " + res.sum;

    updateGraph(res.probs, null, false);
  }

  else if (currentMode === "word") {
    divOut.textContent = res.word || "-";

    document.getElementById("prob").innerHTML = "Detected Word";

    document.getElementById("finalResult").innerText =
      "Detected word: " + (res.word || "No letters detected");

    updateGraph(res.probs, null, true);
  }
}

function initProbGraph() {
  svgGraph = d3.select("#probGraph")
    .attr("width", 250)
    .attr("height", 220)
    .append("g");

  updateGraph([], null, false);
}

function updateGraph(probs, predIndex, isLetterMode) {
  if (!svgGraph) return;

  svgGraph.selectAll("*").remove();

  if (!probs || probs.length === 0) return;

  const labels = isLetterMode ? "ABCDEFGHIJKLMNOPQRSTUVWXYZ" : "0123456789";
  const count = isLetterMode ? 26 : 10;
  const barHeight = isLetterMode ? 7 : 14;
  const gap = isLetterMode ? 1 : 5;

  for (let i = 0; i < count; i++) {
    const y = i * (barHeight + gap);
    const value = probs[i] || 0;

    svgGraph.append("text")
      .attr("x", 0)
      .attr("y", y + barHeight)
      .attr("font-size", isLetterMode ? "9" : "12")
      .attr("fill", "#374151")
      .text(labels[i]);

    svgGraph.append("rect")
      .attr("x", 25)
      .attr("y", y)
      .attr("height", barHeight)
      .attr("width", value * 2)
      .attr("rx", 4)
      .style("fill", i === predIndex ? "blue" : "#7c3aed");

    svgGraph.append("text")
      .attr("x", 30 + value * 2)
      .attr("y", y + barHeight)
      .attr("font-size", isLetterMode ? "8" : "10")
      .attr("fill", "#374151")
      .text(value.toFixed(1) + "%");
  }
}