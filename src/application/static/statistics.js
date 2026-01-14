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
  const visualToggleSkeleton = document.getElementById("visualToggleSkeleton");
  const visualToggleAugmented = document.getElementById("visualToggleAugmented");
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

  const axisColors = {
    x: "#2563eb",
    y: "#0b5d5b",
    z: "#16a34a",
  };

  const highlightColors = [lineColors.x, lineColors.y, lineColors.z];

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
  const plotSkeletonElement = document.getElementById("plot-data-skeleton");
  const plotAugmentedElement = document.getElementById("plot-data-augmented");
  const plotSkeletonData = plotSkeletonElement
    ? JSON.parse(plotSkeletonElement.textContent)
    : null;
  const plotAugmentedData = plotAugmentedElement
    ? JSON.parse(plotAugmentedElement.textContent)
    : null;
  const plotContainer = document.getElementById("statsPlot3d");
  const plotEmpty = document.getElementById("statsPlotEmpty");

  const plotState = {
    ready: false,
    loading: false,
    hasData: false,
    cameraByMode: {},
    relayoutBound: false,
    mode: "video",
    data: null,
    activeChartIndex: 0,
  };

  let plotPlaying = false;
  let plotTimer = null;
  let plotIndex = 0;

  const defaultPlotCamera = {
    eye: { x: 0, y: 0, z: 3 },
    up: { x: 0, y: 0, z: 0 },
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
    const current = plotState.cameraByMode[plotState.mode] || defaultPlotCamera;
    const nextCamera = {
      center: current.center,
      eye: rotateVector(current.eye, axis, angle),
      up: rotateVector(current.up, axis, angle),
    };
    plotState.cameraByMode[plotState.mode] = nextCamera;
    if (plotState.ready && plotContainer && window.Plotly) {
      window.Plotly.relayout(plotContainer, { "scene.camera": nextCamera });
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
  const invertPlotValue = (value) => (Number.isFinite(value) ? -value : value);
  const transformPlotPoint = (point) => {
    const x = normalizePlotValue(point?.[0]);
    const y = normalizePlotValue(point?.[1]);
    const z = normalizePlotValue(point?.[2]);
    return {
      x,
      y,
      z,
    };
  };

  const normalizeMarkerName = (value) =>
    value.toLowerCase().replace(/[^a-z0-9]/g, "");

  const resolveHighlightMarkerIndex = () => {
    if (!plotState.data || !plotState.data.markers) {
      return null;
    }
    const index = plotState.activeChartIndex;
    const select = selects[index];
    if (!select) {
      return null;
    }
    const value = select.value || "";
    if (!value.startsWith("marker:")) {
      return null;
    }
    const markerName = normalizeMarkerName(value.slice("marker:".length));
    const markers = plotState.data.markers;
    const resolvedIndex = markers.findIndex(
      (marker) => normalizeMarkerName(String(marker)) === markerName
    );
    return resolvedIndex >= 0 ? resolvedIndex : null;
  };

  const buildPlotTraces = (frame) => {
    const points = frame || [];
    const transformed = points.map(transformPlotPoint);
    const x = transformed.map((point) => point.x);
    const y = transformed.map((point) => point.y);
    const z = transformed.map((point) => point.z);
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

    const connections = plotState.data?.connections || [];
    connections.forEach(([start, end]) => {
      const startPoint = transformPlotPoint(points[start]);
      const endPoint = transformPlotPoint(points[end]);
      traces.push({
        type: "scatter3d",
        mode: "lines",
        x: [startPoint?.x, endPoint?.x],
        y: [startPoint?.y, endPoint?.y],
        z: [startPoint?.z, endPoint?.z],
        line: { color: "#2563eb", width: 4 },
        hoverinfo: "skip",
      });
    });

    const highlightIndex = resolveHighlightMarkerIndex();
    if (highlightIndex !== null && points[highlightIndex]) {
      const highlightPoint = transformPlotPoint(points[highlightIndex]);
      traces.push({
        type: "scatter3d",
        mode: "markers",
        x: [highlightPoint.x],
        y: [highlightPoint.y],
        z: [highlightPoint.z],
        marker: {
          size: 7,
          color: highlightColors[plotState.activeChartIndex] || "#2563eb",
        },
        hoverinfo: "skip",
      });
    }
    return traces;
  };

  const resolvePlotIndex = (timeValue) => {
    const frames = plotState.data?.frames || [];
    if (!frames.length) {
      return 0;
    }
    if (timelineDuration <= 0) {
      return 0;
    }
    const ratio = clampTime(timeValue - timelineStart) / timelineDuration;
    return Math.min(Math.round(ratio * (frames.length - 1)), frames.length - 1);
  };

  const renderPlotFrame = (timeValue) => {
    if (!plotState.hasData || !plotContainer || !window.Plotly || !plotState.data) {
      return;
    }
    const index = resolvePlotIndex(timeValue);
    plotIndex = index;
    const frame = plotState.data.frames[index] || [];
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
        xaxis: {
          title: { text: "X", font: { color: axisColors.x } },
          tickfont: { color: axisColors.x },
          range: [-1, 1],
          linecolor: axisColors.x,
          linewidth: 2,
          gridcolor: "rgba(37, 99, 235, 0.12)",
        },
        yaxis: {
          title: { text: "Y", font: { color: axisColors.y } },
          tickfont: { color: axisColors.y },
          range: [-1, 1],
          linecolor: axisColors.y,
          linewidth: 2,
          gridcolor: "rgba(11, 93, 91, 0.12)",
        },
        zaxis: {
          title: { text: "Z", font: { color: axisColors.z } },
          tickfont: { color: axisColors.z },
          range: [-1, 1],
          linecolor: axisColors.z,
          linewidth: 2,
          gridcolor: "rgba(22, 163, 74, 0.12)",
        },
        aspectmode: "cube",
        camera: plotState.cameraByMode[plotState.mode] || defaultPlotCamera,
      },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
    };
    const config = { displayModeBar: false, responsive: true };
    window.Plotly.react(plotContainer, buildPlotTraces(frame), layout, config).then(() => {
      if (!plotState.relayoutBound) {
        plotContainer.on("plotly_relayout", (event) => {
          if (event?.scene?.camera) {
            plotState.cameraByMode[plotState.mode] = event.scene.camera;
            return;
          }
          if (event && event["scene.camera"]) {
            plotState.cameraByMode[plotState.mode] = event["scene.camera"];
          }
        });
        plotState.relayoutBound = true;
      }
    });
    plotState.ready = true;
  };

  const resolvePlotTimeFromIndex = (index) => {
    const frames = plotState.data?.frames || [];
    if (!frames.length) {
      return timelineStart;
    }
    const safeIndex = Math.max(0, Math.min(index, frames.length - 1));
    const times = plotState.data?.times || [];
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
    const frames = plotState.data?.frames || [];
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
    const times = plotState.data?.times || [];
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

  const resolvePlotData = (mode) => {
    if (mode === "skeleton") {
      return plotSkeletonData;
    }
    if (mode === "augmented") {
      return plotAugmentedData;
    }
    return null;
  };

  const setActiveVisual = (mode) => {
    if (
      !statsVideoPane
      || !statsPlotPane
      || !visualToggleVideo
      || !visualToggleSkeleton
      || !visualToggleAugmented
    ) {
      return;
    }
    const isVideo = mode === "video";
    plotState.mode = mode;
    plotState.data = resolvePlotData(mode);
    plotState.hasData = !!(plotState.data && plotState.data.frames?.length);
    plotState.ready = false;
    plotIndex = 0;
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
    if (plotContainer && plotEmpty) {
      plotContainer.parentElement?.classList.toggle("d-none", !plotState.hasData);
      plotEmpty.classList.toggle("d-none", plotState.hasData);
    }
    visualToggleVideo.classList.toggle("btn-primary", isVideo);
    visualToggleVideo.classList.toggle("btn-outline-primary", !isVideo);
    visualToggleSkeleton.classList.toggle("btn-primary", mode === "skeleton");
    visualToggleSkeleton.classList.toggle("btn-outline-primary", mode !== "skeleton");
    visualToggleAugmented.classList.toggle("btn-primary", mode === "augmented");
    visualToggleAugmented.classList.toggle("btn-outline-primary", mode !== "augmented");
    if (statsVisualTitle) {
      if (mode === "skeleton") {
        statsVisualTitle.textContent = "Skeleton plot";
      } else if (mode === "augmented") {
        statsVisualTitle.textContent = "Augmented plot";
      } else {
        statsVisualTitle.textContent = "Video frame";
      }
    }
    if (isVideo) {
      stopPlotPlayback();
    }
    if (!isVideo && plotState.hasData && plotContainer && !plotState.ready) {
      loadPlotly()
        .then(() => {
          const currentValue = Number.parseFloat(slider.value || "0");
          plotState.relayoutBound = false;
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

  const chartCards = Array.from(document.querySelectorAll(".stats-chart-card"));

  const setActiveChart = (index) => {
    plotState.activeChartIndex = index;
    chartCards.forEach((card) => {
      const cardIndex = Number.parseInt(card.dataset.chartIndex || "-1", 10);
      card.classList.toggle("is-active", cardIndex === index);
    });
    if (plotState.ready) {
      const timeValue = timelineStart + Number.parseFloat(slider.value || "0");
      renderPlotFrame(timeValue);
    }
  };

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

  if (visualToggleVideo && visualToggleSkeleton && visualToggleAugmented) {
    visualToggleVideo.addEventListener("click", () => {
      setActiveVisual("video");
    });
    visualToggleSkeleton.addEventListener("click", () => {
      setActiveVisual("skeleton");
    });
    visualToggleAugmented.addEventListener("click", () => {
      setActiveVisual("augmented");
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
      plotState.cameraByMode[plotState.mode] = null;
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
      if (plotState.ready) {
        const timeValue = timelineStart + Number.parseFloat(slider.value || "0");
        renderPlotFrame(timeValue);
      }
    });
  });

  chartCards.forEach((card) => {
    const cardIndex = Number.parseInt(card.dataset.chartIndex || "-1", 10);
    if (cardIndex < 0) {
      return;
    }
    card.addEventListener("click", () => {
      setActiveChart(cardIndex);
    });
  });

  let initialVisual = "video";
  try {
    const storedVisual = localStorage.getItem(visualStorageKey);
    if (storedVisual === "plot") {
      initialVisual = "augmented";
    } else if (
      storedVisual === "skeleton"
      || storedVisual === "augmented"
      || storedVisual === "video"
    ) {
      initialVisual = storedVisual;
    }
  } catch (error) {
    // Ignore storage failures (private mode or quota).
  }

  setActiveVisual(initialVisual);
  setActiveChart(plotState.activeChartIndex);
  // No select styling; highlight follows active chart.
  setSliderTime(0, false);
});
