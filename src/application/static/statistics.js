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

  setSliderTime(0, false);
});
