document.addEventListener("DOMContentLoaded", () => {
  const markerList = document.getElementById("marker-list");
  const seriesList = document.getElementById("marker-series");
  if (!markerList || !seriesList) {
    return;
  }

  const markers = JSON.parse(markerList.textContent);
  const series = JSON.parse(seriesList.textContent);
  if (!markers.length) {
    return;
  }

  const previewVideo = document.getElementById("previewVideo");
  const slider = document.getElementById("timelineSlider");
  const playButton = document.getElementById("timelinePlay");
  const playIcon = document.getElementById("timelinePlayIcon");
  const playLabel = document.getElementById("timelinePlayLabel");
  const timeLabel = document.getElementById("timelineTime");
  const durationLabel = document.getElementById("timelineDuration");
  const plotControls = document.getElementById("plotControls");
  const plotPlayButton = document.getElementById("plotPlay");
  const plotPlayIcon = document.getElementById("plotPlayIcon");
  const plotPlayLabel = document.getElementById("plotPlayLabel");
  const plotRotateLeft = document.getElementById("plotRotateLeft");
  const plotRotateRight = document.getElementById("plotRotateRight");
  const plotRotateUp = document.getElementById("plotRotateUp");
  const plotRotateDown = document.getElementById("plotRotateDown");
  const plotRotateRollLeft = document.getElementById("plotRotateRollLeft");
  const plotRotateRollRight = document.getElementById("plotRotateRollRight");
  const plotRotateReset = document.getElementById("plotRotateReset");
  const visualToggleVideo = document.getElementById("visualToggleVideo");
  const visualTogglePlot = document.getElementById("visualTogglePlot");
  const statsVideoPane = document.getElementById("statsVideoPane");
  const statsPlotPane = document.getElementById("statsPlotPane");
  const statsVisualTitle = document.getElementById("statsVisualTitle");
  if (!slider || !playButton || !playIcon || !playLabel || !timeLabel || !durationLabel) {
    return;
  }

  const chartValueSets = [
    {
      x: document.getElementById("chartAValueX"),
      y: document.getElementById("chartAValueY"),
      z: document.getElementById("chartAValueZ"),
    },
    {
      x: document.getElementById("chartBValueX"),
      y: document.getElementById("chartBValueY"),
      z: document.getElementById("chartBValueZ"),
    },
    {
      x: document.getElementById("chartCValueX"),
      y: document.getElementById("chartCValueY"),
      z: document.getElementById("chartCValueZ"),
    },
  ];

  const lineColors = {
    x: "#9ad9ff",
    y: "#0b5d5b",
    z: "#2effb3",
  };

  const timeline = Object.values(series).reduce((best, entry) => {
    if (entry && Array.isArray(entry.t) && entry.t.length > best.length) {
      return entry.t;
    }
    return best;
  }, []);

  const estimateStep = (values) => {
    if (!values || values.length < 2) {
      return 0.033;
    }
    const deltas = [];
    for (let i = 1; i < values.length; i += 1) {
      const delta = values[i] - values[i - 1];
      if (Number.isFinite(delta) && delta > 0) {
        deltas.push(delta);
      }
    }
    if (!deltas.length) {
      return 0.033;
    }
    const sum = deltas.reduce((total, value) => total + value, 0);
    return sum / deltas.length;
  };

  const timelineStart = timeline.length ? timeline[0] : 0;
  const timelineEnd = timeline.length ? timeline[timeline.length - 1] : 0;
  const timelineDuration = Math.max(timelineEnd - timelineStart, 0);
  const stepSeconds = estimateStep(timeline);
  slider.min = "0";
  slider.max = String(timelineDuration);
  slider.step = String(stepSeconds);
  slider.value = "0";
  durationLabel.textContent = timelineDuration ? `${timelineDuration.toFixed(2)}s` : "0.00s";

  const selects = [
    document.getElementById("markerSelectA"),
    document.getElementById("markerSelectB"),
    document.getElementById("markerSelectC"),
  ];

  const runKey = document.body?.dataset?.runKey || "default";
  const selectionStorageKey = `kinetiq:stats:selection:${runKey}`;
  const visualStorageKey = `kinetiq:stats:visual:${runKey}`;
  const plotDataElement = document.getElementById("plot-data");
  const plotData = plotDataElement ? JSON.parse(plotDataElement.textContent) : null;
  const plotContainer = document.getElementById("statsPlot3d");

  const plotState = {
    ready: false,
    loading: false,
    hasData: !!(plotData && plotData.frames && plotData.frames.length),
    camera: null,
    relayoutBound: false,
  };

  let plotPlaying = false;
  let plotTimer = null;
  let plotIndex = 0;

  const defaultPlotCamera = {
    eye: { x: 0, y: 0, z: 2.4 },
    up: { x: -1, y: 0, z: 0 },
    center: { x: 0, y: 0, z: 0 },
  };

  const rotateVector = (vector, axis, angle) => {
    const { x, y, z } = vector;
    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);
    if (axis === "x") {
      return {
        x,
        y: y * cosA - z * sinA,
        z: y * sinA + z * cosA,
      };
    }
    if (axis === "y") {
      return {
        x: x * cosA + z * sinA,
        y,
        z: -x * sinA + z * cosA,
      };
    }
    return {
      x: x * cosA - y * sinA,
      y: x * sinA + y * cosA,
      z,
    };
  };

  const applyCameraRotation = (axis, angle) => {
    const current = plotState.camera || defaultPlotCamera;
    plotState.camera = {
      center: current.center,
      eye: rotateVector(current.eye, axis, angle),
      up: rotateVector(current.up, axis, angle),
    };
    if (plotState.ready && plotContainer && window.Plotly) {
      window.Plotly.relayout(plotContainer, { "scene.camera": plotState.camera });
    }
  };

  const resolveAxisRange = (range, invert = false) => {
    if (!range || range.length !== 2) {
      return undefined;
    }
    return invert ? [range[1], range[0]] : range;
  };

  const loadPlotly = () => {
    if (window.Plotly) {
      return Promise.resolve(window.Plotly);
    }
    if (plotState.loading) {
      return new Promise((resolve, reject) => {
        const waitForPlotly = () => {
          if (window.Plotly) {
            resolve(window.Plotly);
          } else if (plotState.loading) {
            requestAnimationFrame(waitForPlotly);
          } else {
            reject(new Error("Plotly failed to load."));
          }
        };
        waitForPlotly();
      });
    }
    plotState.loading = true;
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "https://cdn.plot.ly/plotly-2.27.0.min.js";
      script.async = true;
      script.onload = () => {
        plotState.loading = false;
        resolve(window.Plotly);
      };
      script.onerror = () => {
        plotState.loading = false;
        reject(new Error("Plotly failed to load."));
      };
      document.head.appendChild(script);
    });
  };

  const normalizePlotValue = (value) => (Number.isFinite(value) ? value : null);

  const buildPlotTraces = (frame) => {
    const points = frame || [];
    const x = points.map((point) => normalizePlotValue(point?.[0]));
    const y = points.map((point) => normalizePlotValue(point?.[1]));
    const z = points.map((point) => normalizePlotValue(point?.[2]));
    const traces = [
      {
        type: "scatter3d",
        mode: "markers",
        x,
        y,
        z,
        marker: { size: 3, color: "#0b5d5b" },
        hoverinfo: "skip",
      },
    ];

    const connections = plotData?.connections || [];
    connections.forEach(([start, end]) => {
      const startPoint = points[start];
      const endPoint = points[end];
      traces.push({
        type: "scatter3d",
        mode: "lines",
        x: [normalizePlotValue(startPoint?.[0]), normalizePlotValue(endPoint?.[0])],
        y: [normalizePlotValue(startPoint?.[1]), normalizePlotValue(endPoint?.[1])],
        z: [normalizePlotValue(startPoint?.[2]), normalizePlotValue(endPoint?.[2])],
        line: { color: "#2563eb", width: 4 },
        hoverinfo: "skip",
      });
    });
    return traces;
  };

  const resolvePlotIndex = (timeValue) => {
    const frames = plotData?.frames || [];
    if (!frames.length) {
      return 0;
    }
    const times = plotData?.times || [];
    const hasTimes =
      times.length === frames.length && times.some((value) => Number.isFinite(value));
    if (hasTimes) {
      let bestIndex = 0;
      let bestDiff = Infinity;
      times.forEach((value, index) => {
        if (!Number.isFinite(value)) {
          return;
        }
        const diff = Math.abs(value - timeValue);
        if (diff < bestDiff) {
          bestDiff = diff;
          bestIndex = index;
        }
      });
      return bestIndex;
    }
    if (timelineDuration <= 0) {
      return 0;
    }
    const ratio = clampTime(timeValue - timelineStart) / timelineDuration;
    return Math.min(Math.round(ratio * (frames.length - 1)), frames.length - 1);
  };

  const renderPlotFrame = (timeValue) => {
    if (!plotState.hasData || !plotContainer || !window.Plotly) {
      return;
    }
    const index = resolvePlotIndex(timeValue);
    plotIndex = index;
    const frame = plotData.frames[index] || [];
    const width = plotContainer.clientWidth || undefined;
    const height = plotContainer.clientHeight || undefined;
    const layout = {
      margin: { l: 0, r: 0, t: 0, b: 0 },
      width,
      height,
      autosize: true,
      showlegend: false,
      dragmode: "orbit",
      scene: {
        xaxis: { title: "X", range: [-1, 1] },
        yaxis: { title: "Y", range: [1, -1] },
        zaxis: { title: "Z", range: [-1, 1] },
        aspectmode: "cube",
        camera: plotState.camera || defaultPlotCamera,
      },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
    };
    const config = { displayModeBar: false, responsive: true };
    window.Plotly.react(plotContainer, buildPlotTraces(frame), layout, config).then(() => {
      if (!plotState.relayoutBound) {
        plotContainer.on("plotly_relayout", (event) => {
          if (event?.scene?.camera) {
            plotState.camera = event.scene.camera;
            return;
          }
          if (event && event["scene.camera"]) {
            plotState.camera = event["scene.camera"];
          }
        });
        plotState.relayoutBound = true;
      }
    });
    plotState.ready = true;
  };

  const resolvePlotTimeFromIndex = (index) => {
    const frames = plotData?.frames || [];
    if (!frames.length) {
      return timelineStart;
    }
    const safeIndex = Math.max(0, Math.min(index, frames.length - 1));
    const times = plotData?.times || [];
    if (times.length === frames.length && Number.isFinite(times[safeIndex])) {
      return times[safeIndex];
    }
    if (timelineDuration <= 0 || frames.length === 1) {
      return timelineStart;
    }
    const ratio = safeIndex / (frames.length - 1);
    return timelineStart + ratio * timelineDuration;
  };

  const stopPlotPlayback = () => {
    plotPlaying = false;
    if (plotPlayIcon) {
      plotPlayIcon.className = "bi bi-play-fill";
    }
    if (plotPlayLabel) {
      plotPlayLabel.textContent = "Play";
    }
    if (plotTimer) {
      clearInterval(plotTimer);
      plotTimer = null;
    }
  };

  const startPlotPlayback = () => {
    const frames = plotData?.frames || [];
    if (!frames.length) {
      return;
    }
    plotPlaying = true;
    if (plotPlayIcon) {
      plotPlayIcon.className = "bi bi-pause-fill";
    }
    if (plotPlayLabel) {
      plotPlayLabel.textContent = "Pause";
    }
    const times = plotData?.times || [];
    const step = estimateStep(times);
    const intervalMs = step > 0 ? step * 1000 : 33;
    plotTimer = setInterval(() => {
      const nextIndex = plotIndex + 1;
      if (nextIndex >= frames.length) {
        stopPlotPlayback();
        return;
      }
      plotIndex = nextIndex;
      renderPlotFrame(resolvePlotTimeFromIndex(plotIndex));
    }, intervalMs);
  };

  const setActiveVisual = (mode) => {
    if (!statsVideoPane || !statsPlotPane || !visualToggleVideo || !visualTogglePlot) {
      return;
    }
    const isVideo = mode === "video";
    try {
      localStorage.setItem(visualStorageKey, mode);
    } catch (error) {
      // Ignore storage failures (private mode or quota).
    }
    statsVideoPane.classList.toggle("d-none", !isVideo);
    statsPlotPane.classList.toggle("d-none", isVideo);
    if (plotControls) {
      plotControls.classList.toggle("d-none", isVideo || !plotState.hasData);
    }
    visualToggleVideo.classList.toggle("btn-primary", isVideo);
    visualToggleVideo.classList.toggle("btn-outline-primary", !isVideo);
    visualTogglePlot.classList.toggle("btn-primary", !isVideo);
    visualTogglePlot.classList.toggle("btn-outline-primary", isVideo);
    if (statsVisualTitle) {
      statsVisualTitle.textContent = isVideo ? "Video frame" : "3D plot";
    }
    if (isVideo) {
      stopPlotPlayback();
    }
    if (!isVideo && plotState.hasData && plotContainer && !plotState.ready) {
      loadPlotly()
        .then(() => {
          const currentValue = Number.parseFloat(slider.value || "0");
          renderPlotFrame(timelineStart + currentValue);
        })
        .catch(() => {
          // Plot container remains empty if Plotly fails to load.
        });
    }
  };

  const persistSelection = () => {
    const values = selects.map((select) => select.value || "");
    try {
      localStorage.setItem(selectionStorageKey, JSON.stringify({ values }));
    } catch (error) {
      // Ignore storage failures (private mode or quota).
    }
  };

  const loadPersistedSelection = () => {
    try {
      const stored = localStorage.getItem(selectionStorageKey);
      if (!stored) {
        return null;
      }
      const parsed = JSON.parse(stored);
      if (!parsed || !Array.isArray(parsed.values)) {
        return null;
      }
      return parsed.values.slice(0, selects.length);
    } catch (error) {
      return null;
    }
  };

  const resolveMarker = (value) => markers.find((item) => item.value === value) || null;
  const persistedSelection = loadPersistedSelection();
  const defaultMarkers = persistedSelection
    ? persistedSelection.map(resolveMarker)
    : [null, null, null];

  const activeMarkerValues = defaultMarkers.map((marker) => marker?.value || "");

  const createOptions = (select, selectedValue) => {
    select.innerHTML = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "None";
    placeholder.selected = !selectedValue;
    select.appendChild(placeholder);
    markers.forEach((marker) => {
      const option = document.createElement("option");
      option.value = marker.value;
      option.textContent = marker.label;
      option.selected = selectedValue && marker.value === selectedValue.value;
      select.appendChild(option);
    });
  };

  selects.forEach((select, index) => {
    createOptions(select, defaultMarkers[index]);
    select.value = defaultMarkers[index]?.value || "";
  });

  const buildDataset = (marker) => {
    if (!marker) {
      return { labels: [], datasets: [] };
    }
    const data = series[marker.value];
    const points = (values) => data.t.map((t, i) => ({ x: t, y: values[i] ?? null }));
    return {
      datasets: [
        {
          label: data.labels[0] || "X",
          data: points(data.x),
          borderColor: lineColors.x,
          backgroundColor: lineColors.x,
          borderWidth: 2,
        },
        {
          label: data.labels[1] || "Y",
          data: points(data.y),
          borderColor: lineColors.y,
          backgroundColor: lineColors.y,
          borderWidth: 2,
        },
        {
          label: data.labels[2] || "Z",
          data: points(data.z),
          borderColor: lineColors.z,
          backgroundColor: lineColors.z,
          borderWidth: 2,
        },
      ],
    };
  };

  const findClosestIndex = (values, target) => {
    if (!values.length) {
      return -1;
    }
    let bestIndex = 0;
    let bestDiff = Math.abs(values[0] - target);
    for (let i = 1; i < values.length; i += 1) {
      const diff = Math.abs(values[i] - target);
      if (diff < bestDiff) {
        bestDiff = diff;
        bestIndex = i;
      }
    }
    return bestIndex;
  };

  const formatValue = (value) => (Number.isFinite(value) ? value.toFixed(2) : "--");

  const updateValueSet = (index, timeValue) => {
    const valueTargets = chartValueSets[index];
    if (!valueTargets) {
      return;
    }
    const markerValue = activeMarkerValues[index];
    const seriesData = markerValue ? series[markerValue] : null;
    if (!seriesData || !Array.isArray(seriesData.t) || !seriesData.t.length) {
      valueTargets.x.textContent = "X: --";
      valueTargets.y.textContent = "Y: --";
      valueTargets.z.textContent = "Z: --";
      return;
    }
    const indexAtTime = findClosestIndex(seriesData.t, timeValue);
    if (indexAtTime < 0) {
      valueTargets.x.textContent = "X: --";
      valueTargets.y.textContent = "Y: --";
      valueTargets.z.textContent = "Z: --";
      return;
    }
    const labels = Array.isArray(seriesData.labels) ? seriesData.labels : [];
    const labelX = labels[0] || "X";
    const labelY = labels[1] || "Y";
    const labelZ = labels[2] || "Z";
    const valueX = seriesData.x?.[indexAtTime];
    const valueY = seriesData.y?.[indexAtTime];
    const valueZ = seriesData.z?.[indexAtTime];
    valueTargets.x.textContent = `${labelX}: ${formatValue(valueX)}`;
    valueTargets.y.textContent = `${labelY}: ${formatValue(valueY)}`;
    valueTargets.z.textContent = `${labelZ}: ${formatValue(valueZ)}`;
  };

  const playheadPlugin = {
    id: "playheadLine",
    afterDatasetsDraw(chart, args, options) {
      const timeValue = options?.time;
      if (!Number.isFinite(timeValue)) {
        return;
      }
      const xScale = chart.scales.x;
      if (!xScale) {
        return;
      }
      const x = xScale.getPixelForValue(timeValue);
      if (!Number.isFinite(x)) {
        return;
      }
      const { ctx, chartArea } = chart;
      ctx.save();
      ctx.strokeStyle = options?.color || "rgba(15, 23, 42, 0.6)";
      ctx.lineWidth = options?.width || 2;
      if (options?.dash) {
        ctx.setLineDash(options.dash);
      }
      ctx.beginPath();
      ctx.moveTo(x, chartArea.top);
      ctx.lineTo(x, chartArea.bottom);
      ctx.stroke();
      ctx.restore();
    },
  };

  Chart.register(playheadPlugin);

  const buildChart = (canvasId, marker) => {
    const ctx = document.getElementById(canvasId);
    return new Chart(ctx, {
      type: "line",
      data: buildDataset(marker),
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        elements: {
          point: {
            radius: 0,
            hoverRadius: 0,
          },
        },
        scales: {
          x: {
            type: "linear",
            min: timelineStart,
            max: timelineEnd,
            title: { display: true, text: "Time (s)" },
          },
          y: { title: { display: true, text: "Angle (deg)" } },
        },
        plugins: {
          legend: { display: false },
          playheadLine: {
            time: timeline.length ? timeline[0] : 0,
            color: "rgba(15, 23, 42, 0.55)",
            width: 2,
          },
        },
      },
    });
  };

  const charts = [
    buildChart("chartA", defaultMarkers[0]),
    buildChart("chartB", defaultMarkers[1]),
    buildChart("chartC", defaultMarkers[2]),
  ];

  const updateChartsPlayhead = (timeValue) => {
    charts.forEach((chart, index) => {
      chart.options.plugins.playheadLine.time = timeValue;
      chart.update("none");
      updateValueSet(index, timeValue);
    });
  };

  const clampTime = (value) => Math.max(0, Math.min(value, timelineDuration));

  let videoTimeScale = 1;

  const setSliderTime = (value, updateVideo = true) => {
    const safeTime = clampTime(value);
    slider.value = String(safeTime);
    const timeValue = timelineStart + safeTime;
    timeLabel.textContent = `${timeValue.toFixed(2)}s`;
    updateChartsPlayhead(timeValue);
    if (plotState.ready) {
      renderPlotFrame(timeValue);
    }
    if (previewVideo && updateVideo) {
      previewVideo.currentTime = safeTime * videoTimeScale;
    }
  };

  let playing = false;
  let fallbackTimer = null;
  let rafId = null;

  const stopPlayback = () => {
    playing = false;
    playIcon.className = "bi bi-play-fill";
    playLabel.textContent = "Play";
    if (previewVideo) {
      previewVideo.pause();
    }
    if (fallbackTimer) {
      clearInterval(fallbackTimer);
      fallbackTimer = null;
    }
    if (rafId) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
  };

  const startPlayback = () => {
    if (timelineDuration <= 0) {
      return;
    }
    playing = true;
    playIcon.className = "bi bi-pause-fill";
    playLabel.textContent = "Pause";
    if (previewVideo) {
      previewVideo.play().catch(() => {
        stopPlayback();
      });
    } else {
      fallbackTimer = setInterval(() => {
        setSliderTime(Number.parseFloat(slider.value) + stepSeconds);
        if (Number.parseFloat(slider.value) >= timelineDuration) {
          stopPlayback();
        }
      }, stepSeconds * 1000);
    }
    const animate = () => {
      if (!playing) {
        return;
      }
      if (previewVideo) {
        const timeValue = previewVideo.currentTime / videoTimeScale;
        setSliderTime(timeValue, false);
      }
      rafId = requestAnimationFrame(animate);
    };
    rafId = requestAnimationFrame(animate);
  };

  slider.addEventListener("input", () => {
    stopPlayback();
    setSliderTime(Number.parseFloat(slider.value), true);
  });

  playButton.addEventListener("click", () => {
    if (playing) {
      stopPlayback();
    } else {
      startPlayback();
    }
  });

  if (previewVideo) {
    previewVideo.addEventListener("loadedmetadata", () => {
      if (timelineDuration > 0 && previewVideo.duration > 0) {
        videoTimeScale = previewVideo.duration / timelineDuration;
      }
    });
    previewVideo.addEventListener("pause", () => {
      if (playing) {
        stopPlayback();
      }
    });
  }

  if (visualToggleVideo && visualTogglePlot) {
    visualToggleVideo.addEventListener("click", () => {
      setActiveVisual("video");
    });
    visualTogglePlot.addEventListener("click", () => {
      setActiveVisual("plot");
    });
  }

  if (plotPlayButton) {
    plotPlayButton.addEventListener("click", () => {
      if (plotPlaying) {
        stopPlotPlayback();
      } else {
        startPlotPlayback();
      }
    });
  }

  const rotationStep = Math.PI / 12;
  if (plotRotateLeft) {
    plotRotateLeft.addEventListener("click", () => {
      applyCameraRotation("z", rotationStep);
    });
  }
  if (plotRotateRight) {
    plotRotateRight.addEventListener("click", () => {
      applyCameraRotation("z", -rotationStep);
    });
  }
  if (plotRotateUp) {
    plotRotateUp.addEventListener("click", () => {
      applyCameraRotation("x", rotationStep);
    });
  }
  if (plotRotateDown) {
    plotRotateDown.addEventListener("click", () => {
      applyCameraRotation("x", -rotationStep);
    });
  }
  if (plotRotateRollLeft) {
    plotRotateRollLeft.addEventListener("click", () => {
      applyCameraRotation("y", rotationStep);
    });
  }
  if (plotRotateRollRight) {
    plotRotateRollRight.addEventListener("click", () => {
      applyCameraRotation("y", -rotationStep);
    });
  }
  if (plotRotateReset) {
    plotRotateReset.addEventListener("click", () => {
      plotState.camera = null;
      if (plotState.ready && plotContainer && window.Plotly) {
        window.Plotly.relayout(plotContainer, { "scene.camera": defaultPlotCamera });
      }
    });
  }

  selects.forEach((select, index) => {
    select.addEventListener("change", () => {
      const marker = markers.find((item) => item.value === select.value);
      activeMarkerValues[index] = marker?.value || "";
      charts[index].data = buildDataset(marker);
      charts[index].update();
      updateValueSet(index, timelineStart + Number.parseFloat(slider.value));
      persistSelection();
    });
  });

  let initialVisual = "video";
  try {
    const storedVisual = localStorage.getItem(visualStorageKey);
    if (storedVisual === "plot" || storedVisual === "video") {
      initialVisual = storedVisual;
    }
  } catch (error) {
    // Ignore storage failures (private mode or quota).
  }

  setActiveVisual(initialVisual);
  setSliderTime(0, false);
});
