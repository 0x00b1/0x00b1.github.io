/**
 * Interactive Figures for Jekyll Blog
 * Distill/colah-style visualizations using D3.js
 */

(function() {
  'use strict';

  // Track initialized figures to prevent double-init
  const initializedFigs = new Set();

  // Common utilities
  const utils = {
    formatNumber: function(n, decimals) {
      if (n === null || n === undefined || isNaN(n)) return '--';
      const d = decimals !== undefined ? decimals : 4;
      return n.toFixed(d);
    },

    showError: function(container, message) {
      const errorDiv = document.createElement('div');
      errorDiv.className = 'figure-error';
      errorDiv.innerHTML = '<div class="figure-error-icon">!</div><div>' + message + '</div>';
      container.innerHTML = '';
      container.appendChild(errorDiv);
    },

    hideLoading: function(figId) {
      const loading = document.getElementById(figId + '-loading');
      if (loading) loading.style.display = 'none';
    },

    showPlots: function(figId, plotIds) {
      plotIds.forEach(function(id) {
        const wrapper = document.getElementById(figId + '-plot-' + id);
        if (wrapper && wrapper.parentElement) {
          wrapper.parentElement.style.display = 'block';
        }
      });
    }
  };

  // Precision simulation utilities
  const precision = {
    // Convert to FP32
    toFP32: function(n) {
      const f32 = new Float32Array(1);
      f32[0] = n;
      return f32[0];
    },

    // Simulate BF16: keep sign + 8-bit exp + 7-bit mantissa (truncate lower 16 bits)
    toBF16: function(n) {
      const f32 = new Float32Array(1);
      const u32 = new Uint32Array(f32.buffer);
      f32[0] = n;
      // Truncate lower 16 bits of mantissa
      u32[0] = u32[0] & 0xFFFF0000;
      return f32[0];
    },

    // Simulate FP16: 1-bit sign + 5-bit exp + 10-bit mantissa
    toFP16: function(n) {
      const f32 = new Float32Array(1);
      f32[0] = n;

      const u32 = new Uint32Array(f32.buffer);
      const bits = u32[0];

      const sign = (bits >> 31) & 1;
      let exp = (bits >> 23) & 0xFF;
      let mant = bits & 0x7FFFFF;

      // FP16 bias is 15, FP32 bias is 127
      let fp16Exp = exp - 127 + 15;

      if (exp === 0) {
        // Zero or denormal -> zero
        return sign ? -0 : 0;
      } else if (exp === 255) {
        // Inf or NaN
        return n;
      } else if (fp16Exp <= 0) {
        // Underflow to zero
        return sign ? -0 : 0;
      } else if (fp16Exp >= 31) {
        // Overflow to infinity
        return sign ? -Infinity : Infinity;
      }

      // Keep top 10 bits of mantissa
      const fp16Mant = mant >> 13;

      // Reconstruct in FP32
      const newExp = fp16Exp - 15 + 127;
      const newBits = (sign << 31) | (newExp << 23) | (fp16Mant << 13);

      const result = new Float32Array(1);
      const resultU32 = new Uint32Array(result.buffer);
      resultU32[0] = newBits;

      return result[0];
    },

    round: function(n, dtype) {
      switch (dtype) {
        case 'bf16': return precision.toBF16(n);
        case 'fp16': return precision.toFP16(n);
        case 'fp32': return precision.toFP32(n);
        default: return n;
      }
    }
  };

  // Figure: Trajectory Fork Viewer
  function initTrajectoryFork(config) {
    const figId = config.figId;
    const container = document.getElementById(figId);
    if (!container) return;

    const margin = { top: 20, right: 20, bottom: 30, left: 50 };
    const width = 600;
    const height = 180;

    let data = null;
    let runA = null;
    let runB = null;

    // Fetch data
    fetch(config.dataUrl)
      .then(function(response) {
        if (!response.ok) throw new Error('Failed to load data');
        return response.json();
      })
      .then(function(json) {
        data = json;
        utils.hideLoading(figId);
        utils.showPlots(figId, ['y', 'diff']);
        initControls();
        render();
      })
      .catch(function(err) {
        utils.hideLoading(figId);
        utils.showError(container.querySelector('.figure-plot-container'),
          'Error loading data: ' + err.message);
      });

    function initControls() {
      const selectA = document.getElementById(figId + '-run-a');
      const selectB = document.getElementById(figId + '-run-b');

      data.runs.forEach(function(run, i) {
        const optA = document.createElement('option');
        optA.value = run.id;
        optA.textContent = run.label;
        selectA.appendChild(optA);

        const optB = document.createElement('option');
        optB.value = run.id;
        optB.textContent = run.label;
        selectB.appendChild(optB);
      });

      // Default to first two runs
      if (data.runs.length >= 2) {
        selectA.value = data.runs[0].id;
        selectB.value = data.runs[1].id;
      }

      selectA.addEventListener('change', render);
      selectB.addEventListener('change', render);
      document.getElementById(figId + '-show-events')
        .addEventListener('change', render);
    }

    function getRun(id) {
      return data.runs.find(function(r) { return r.id === id; });
    }

    function render() {
      const selectA = document.getElementById(figId + '-run-a');
      const selectB = document.getElementById(figId + '-run-b');
      const showEvents = document.getElementById(figId + '-show-events').checked;

      runA = getRun(selectA.value);
      runB = getRun(selectB.value);

      if (!runA || !runB) return;

      // Update legend
      document.getElementById(figId + '-legend-a').textContent = runA.label;
      document.getElementById(figId + '-legend-b').textContent = runB.label;

      // Calculate diff
      const T = Math.min(runA.y.length, runB.y.length);
      const diff = [];
      for (let i = 0; i < T; i++) {
        diff.push(Math.abs(runA.y[i] - runB.y[i]));
      }

      // Compute first_contact if not provided
      if (runA.first_contact === undefined) {
        runA.first_contact = runA.contact ? runA.contact.indexOf(1) : -1;
      }
      if (runB.first_contact === undefined) {
        runB.first_contact = runB.contact ? runB.contact.indexOf(1) : -1;
      }

      // Render plots
      renderPlotY(runA, runB, T, showEvents);
      renderPlotDiff(diff, T, runA, runB, showEvents);
      setupHover(T, runA, runB, diff);
    }

    function renderPlotY(runA, runB, T, showEvents) {
      const svg = d3.select('#' + figId + '-plot-y');
      svg.selectAll('*').remove();

      svg.attr('viewBox', '0 0 ' + width + ' ' + height)
         .attr('preserveAspectRatio', 'xMidYMid meet');

      const g = svg.append('g')
        .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

      const innerWidth = width - margin.left - margin.right;
      const innerHeight = height - margin.top - margin.bottom;

      const xScale = d3.scaleLinear()
        .domain([0, T - 1])
        .range([0, innerWidth]);

      const yExtent = d3.extent(runA.y.concat(runB.y));
      const yScale = d3.scaleLinear()
        .domain([yExtent[0] - 0.1 * (yExtent[1] - yExtent[0]),
                 yExtent[1] + 0.1 * (yExtent[1] - yExtent[0])])
        .range([innerHeight, 0]);

      // Grid
      g.append('g')
        .attr('class', 'grid')
        .call(d3.axisLeft(yScale).tickSize(-innerWidth).tickFormat(''));

      // Axes
      g.append('g')
        .attr('class', 'axis')
        .attr('transform', 'translate(0,' + innerHeight + ')')
        .call(d3.axisBottom(xScale).ticks(8));

      g.append('g')
        .attr('class', 'axis')
        .call(d3.axisLeft(yScale).ticks(5));

      // Axis labels
      g.append('text')
        .attr('class', 'axis-label')
        .attr('x', innerWidth / 2)
        .attr('y', innerHeight + 25)
        .attr('text-anchor', 'middle')
        .text('t');

      g.append('text')
        .attr('class', 'axis-label')
        .attr('transform', 'rotate(-90)')
        .attr('x', -innerHeight / 2)
        .attr('y', -35)
        .attr('text-anchor', 'middle')
        .text('y(t)');

      // Line generator
      const line = d3.line()
        .x(function(d, i) { return xScale(i); })
        .y(function(d) { return yScale(d); });

      // Draw lines
      g.append('path')
        .datum(runA.y.slice(0, T))
        .attr('class', 'line line-a')
        .attr('d', line);

      g.append('path')
        .datum(runB.y.slice(0, T))
        .attr('class', 'line line-b')
        .attr('d', line);

      // Event markers
      if (showEvents) {
        if (runA.first_contact >= 0 && runA.first_contact < T) {
          g.append('line')
            .attr('class', 'event-marker event-marker-a')
            .attr('x1', xScale(runA.first_contact))
            .attr('x2', xScale(runA.first_contact))
            .attr('y1', 0)
            .attr('y2', innerHeight);
        }
        if (runB.first_contact >= 0 && runB.first_contact < T) {
          g.append('line')
            .attr('class', 'event-marker event-marker-b')
            .attr('x1', xScale(runB.first_contact))
            .attr('x2', xScale(runB.first_contact))
            .attr('y1', 0)
            .attr('y2', innerHeight);
        }
      }

      // Hover line
      g.append('line')
        .attr('class', 'hover-line')
        .attr('id', figId + '-hover-y')
        .attr('y1', 0)
        .attr('y2', innerHeight)
        .style('display', 'none');

      // Hover overlay
      g.append('rect')
        .attr('class', 'hover-overlay')
        .attr('id', figId + '-overlay-y')
        .attr('width', innerWidth)
        .attr('height', innerHeight);

      // Store scales for hover
      svg.node().__xScale = xScale;
      svg.node().__yScale = yScale;
      svg.node().__innerWidth = innerWidth;
      svg.node().__innerHeight = innerHeight;
    }

    function renderPlotDiff(diff, T, runA, runB, showEvents) {
      const svg = d3.select('#' + figId + '-plot-diff');
      svg.selectAll('*').remove();

      svg.attr('viewBox', '0 0 ' + width + ' ' + height)
         .attr('preserveAspectRatio', 'xMidYMid meet');

      const g = svg.append('g')
        .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

      const innerWidth = width - margin.left - margin.right;
      const innerHeight = height - margin.top - margin.bottom;

      const xScale = d3.scaleLinear()
        .domain([0, T - 1])
        .range([0, innerWidth]);

      const yMax = d3.max(diff) || 1;
      const yScale = d3.scaleLinear()
        .domain([0, yMax * 1.1])
        .range([innerHeight, 0]);

      // Grid
      g.append('g')
        .attr('class', 'grid')
        .call(d3.axisLeft(yScale).tickSize(-innerWidth).tickFormat(''));

      // Axes
      g.append('g')
        .attr('class', 'axis')
        .attr('transform', 'translate(0,' + innerHeight + ')')
        .call(d3.axisBottom(xScale).ticks(8));

      g.append('g')
        .attr('class', 'axis')
        .call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format('.2e')));

      // Axis labels
      g.append('text')
        .attr('class', 'axis-label')
        .attr('x', innerWidth / 2)
        .attr('y', innerHeight + 25)
        .attr('text-anchor', 'middle')
        .text('t');

      g.append('text')
        .attr('class', 'axis-label')
        .attr('transform', 'rotate(-90)')
        .attr('x', -innerHeight / 2)
        .attr('y', -40)
        .attr('text-anchor', 'middle')
        .text('|Δy(t)|');

      // Line generator
      const line = d3.line()
        .x(function(d, i) { return xScale(i); })
        .y(function(d) { return yScale(d); });

      // Draw line
      g.append('path')
        .datum(diff)
        .attr('class', 'line line-diff')
        .attr('d', line);

      // Event markers
      if (showEvents) {
        if (runA.first_contact >= 0 && runA.first_contact < T) {
          g.append('line')
            .attr('class', 'event-marker event-marker-a')
            .attr('x1', xScale(runA.first_contact))
            .attr('x2', xScale(runA.first_contact))
            .attr('y1', 0)
            .attr('y2', innerHeight);
        }
        if (runB.first_contact >= 0 && runB.first_contact < T) {
          g.append('line')
            .attr('class', 'event-marker event-marker-b')
            .attr('x1', xScale(runB.first_contact))
            .attr('x2', xScale(runB.first_contact))
            .attr('y1', 0)
            .attr('y2', innerHeight);
        }
      }

      // Hover line
      g.append('line')
        .attr('class', 'hover-line')
        .attr('id', figId + '-hover-diff')
        .attr('y1', 0)
        .attr('y2', innerHeight)
        .style('display', 'none');

      // Hover overlay
      g.append('rect')
        .attr('class', 'hover-overlay')
        .attr('id', figId + '-overlay-diff')
        .attr('width', innerWidth)
        .attr('height', innerHeight);

      // Store scales
      svg.node().__xScale = xScale;
      svg.node().__yScale = yScale;
    }

    function setupHover(T, runA, runB, diff) {
      const tooltip = document.getElementById(figId + '-tooltip');
      const overlayY = document.getElementById(figId + '-overlay-y');
      const overlayDiff = document.getElementById(figId + '-overlay-diff');
      const hoverY = document.getElementById(figId + '-hover-y');
      const hoverDiff = document.getElementById(figId + '-hover-diff');
      const svgY = document.getElementById(figId + '-plot-y');
      const svgDiff = document.getElementById(figId + '-plot-diff');

      function handleMove(e) {
        const svg = e.target.closest('svg');
        if (!svg || !svg.__xScale) return;

        const rect = svg.getBoundingClientRect();
        const x = e.clientX - rect.left - margin.left * (rect.width / width);
        const xScaled = x / (rect.width / width);

        const t = Math.round(svg.__xScale.invert(xScaled));
        if (t < 0 || t >= T) {
          hideHover();
          return;
        }

        const xPos = svg.__xScale(t);

        // Show hover lines
        hoverY.setAttribute('x1', xPos);
        hoverY.setAttribute('x2', xPos);
        hoverY.style.display = 'block';

        hoverDiff.setAttribute('x1', xPos);
        hoverDiff.setAttribute('x2', xPos);
        hoverDiff.style.display = 'block';

        // Update tooltip
        const yA = runA.y[t];
        const yB = runB.y[t];
        const d = diff[t];
        const itA = runA.it ? runA.it[t] : null;
        const itB = runB.it ? runB.it[t] : null;
        const cA = runA.contact ? runA.contact[t] : null;
        const cB = runB.contact ? runB.contact[t] : null;

        let html = '<div class="figure-tooltip-row"><span class="figure-tooltip-label">t:</span><span class="figure-tooltip-value">' + t + '</span></div>';
        html += '<div class="figure-tooltip-row"><span class="figure-tooltip-label">y_A:</span><span class="figure-tooltip-value color-a">' + utils.formatNumber(yA) + '</span></div>';
        html += '<div class="figure-tooltip-row"><span class="figure-tooltip-label">y_B:</span><span class="figure-tooltip-value color-b">' + utils.formatNumber(yB) + '</span></div>';
        html += '<div class="figure-tooltip-row"><span class="figure-tooltip-label">|Δy|:</span><span class="figure-tooltip-value">' + utils.formatNumber(d, 6) + '</span></div>';

        if (itA !== null) {
          html += '<div class="figure-tooltip-row"><span class="figure-tooltip-label">it_A:</span><span class="figure-tooltip-value">' + itA + '</span></div>';
        }
        if (itB !== null) {
          html += '<div class="figure-tooltip-row"><span class="figure-tooltip-label">it_B:</span><span class="figure-tooltip-value">' + itB + '</span></div>';
        }
        if (cA !== null) {
          html += '<div class="figure-tooltip-row"><span class="figure-tooltip-label">contact_A:</span><span class="figure-tooltip-value">' + cA + '</span></div>';
        }
        if (cB !== null) {
          html += '<div class="figure-tooltip-row"><span class="figure-tooltip-label">contact_B:</span><span class="figure-tooltip-value">' + cB + '</span></div>';
        }

        tooltip.innerHTML = html;
        tooltip.classList.add('visible');

        // Position tooltip
        const containerRect = container.querySelector('.figure-plot-container').getBoundingClientRect();
        let tooltipX = e.clientX - containerRect.left + 10;
        let tooltipY = e.clientY - containerRect.top - 10;

        // Keep tooltip in bounds
        const tooltipRect = tooltip.getBoundingClientRect();
        if (tooltipX + tooltipRect.width > containerRect.width) {
          tooltipX = e.clientX - containerRect.left - tooltipRect.width - 10;
        }
        if (tooltipY < 0) tooltipY = 10;

        tooltip.style.left = tooltipX + 'px';
        tooltip.style.top = tooltipY + 'px';
      }

      function hideHover() {
        hoverY.style.display = 'none';
        hoverDiff.style.display = 'none';
        tooltip.classList.remove('visible');
      }

      overlayY.addEventListener('mousemove', handleMove);
      overlayDiff.addEventListener('mousemove', handleMove);
      overlayY.addEventListener('mouseleave', hideHover);
      overlayDiff.addEventListener('mouseleave', hideHover);
    }
  }

  // Figure: Iteration-count Discontinuity Explorer
  function initItDiscontinuity(config) {
    const figId = config.figId;
    const container = document.getElementById(figId);
    if (!container) return;

    const margin = { top: 15, right: 20, bottom: 30, left: 55 };
    const width = 600;
    const height = 140;

    let data = null;
    let currentMode = null;

    fetch(config.dataUrl)
      .then(function(response) {
        if (!response.ok) throw new Error('Failed to load data');
        return response.json();
      })
      .then(function(json) {
        data = json;
        utils.hideLoading(figId);
        utils.showPlots(figId, ['it', 'loss', 'grad']);
        initControls();
        render();
      })
      .catch(function(err) {
        utils.hideLoading(figId);
        utils.showError(container.querySelector('.figure-plot-container'),
          'Error loading data: ' + err.message);
      });

    function initControls() {
      const select = document.getElementById(figId + '-mode');

      data.modes.forEach(function(mode) {
        const opt = document.createElement('option');
        opt.value = mode.id;
        opt.textContent = mode.label;
        select.appendChild(opt);
      });

      select.addEventListener('change', render);
    }

    function getMode(id) {
      return data.modes.find(function(m) { return m.id === id; });
    }

    function render() {
      const select = document.getElementById(figId + '-mode');
      currentMode = getMode(select.value);

      if (!currentMode) return;

      // Show/hide finite diff legend
      const fdLegend = document.getElementById(figId + '-legend-fd');
      fdLegend.style.display = currentMode.grad_fd ? 'flex' : 'none';

      const k = currentMode.k;
      const N = k.length;

      renderPlotIt(currentMode, N);
      renderPlotLoss(currentMode, N);
      renderPlotGrad(currentMode, N);
      setupHover(currentMode, N);
    }

    function renderPlotIt(mode, N) {
      const svg = d3.select('#' + figId + '-plot-it');
      svg.selectAll('*').remove();

      svg.attr('viewBox', '0 0 ' + width + ' ' + height)
         .attr('preserveAspectRatio', 'xMidYMid meet');

      const g = svg.append('g')
        .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

      const innerWidth = width - margin.left - margin.right;
      const innerHeight = height - margin.top - margin.bottom;

      const xExtent = d3.extent(mode.k);
      const xScale = d3.scaleLinear()
        .domain(xExtent)
        .range([0, innerWidth]);

      const yExtent = d3.extent(mode.it);
      const yScale = d3.scaleLinear()
        .domain([yExtent[0] - 0.5, yExtent[1] + 0.5])
        .range([innerHeight, 0]);

      // Find iteration boundaries for shading
      const boundaries = [];
      for (let i = 1; i < N; i++) {
        if (mode.it[i] !== mode.it[i - 1]) {
          boundaries.push(i);
        }
      }

      // Draw iteration bands
      let prevIdx = 0;
      boundaries.forEach(function(idx, i) {
        g.append('rect')
          .attr('class', i % 2 === 0 ? 'it-band' : 'it-band-odd')
          .attr('x', xScale(mode.k[prevIdx]))
          .attr('width', xScale(mode.k[idx]) - xScale(mode.k[prevIdx]))
          .attr('y', 0)
          .attr('height', innerHeight);
        prevIdx = idx;
      });
      // Last band
      g.append('rect')
        .attr('class', boundaries.length % 2 === 0 ? 'it-band' : 'it-band-odd')
        .attr('x', xScale(mode.k[prevIdx]))
        .attr('width', xScale(mode.k[N - 1]) - xScale(mode.k[prevIdx]))
        .attr('y', 0)
        .attr('height', innerHeight);

      // Grid
      g.append('g')
        .attr('class', 'grid')
        .call(d3.axisLeft(yScale).tickSize(-innerWidth).tickFormat(''));

      // Axes
      g.append('g')
        .attr('class', 'axis')
        .attr('transform', 'translate(0,' + innerHeight + ')')
        .call(d3.axisBottom(xScale).ticks(8));

      g.append('g')
        .attr('class', 'axis')
        .call(d3.axisLeft(yScale).ticks(4).tickFormat(d3.format('d')));

      // Axis label
      g.append('text')
        .attr('class', 'axis-label')
        .attr('transform', 'rotate(-90)')
        .attr('x', -innerHeight / 2)
        .attr('y', -40)
        .attr('text-anchor', 'middle')
        .text('iterations');

      // Step line
      const stepLine = d3.line()
        .x(function(d, i) { return xScale(mode.k[i]); })
        .y(function(d) { return yScale(d); })
        .curve(d3.curveStepAfter);

      g.append('path')
        .datum(mode.it)
        .attr('class', 'step-line line-it')
        .attr('d', stepLine);

      // Hover line
      g.append('line')
        .attr('class', 'hover-line')
        .attr('id', figId + '-hover-it')
        .attr('y1', 0)
        .attr('y2', innerHeight)
        .style('display', 'none');

      // Hover overlay
      g.append('rect')
        .attr('class', 'hover-overlay')
        .attr('id', figId + '-overlay-it')
        .attr('width', innerWidth)
        .attr('height', innerHeight);

      svg.node().__xScale = xScale;
      svg.node().__innerHeight = innerHeight;
    }

    function renderPlotLoss(mode, N) {
      const svg = d3.select('#' + figId + '-plot-loss');
      svg.selectAll('*').remove();

      svg.attr('viewBox', '0 0 ' + width + ' ' + height)
         .attr('preserveAspectRatio', 'xMidYMid meet');

      const g = svg.append('g')
        .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

      const innerWidth = width - margin.left - margin.right;
      const innerHeight = height - margin.top - margin.bottom;

      const xExtent = d3.extent(mode.k);
      const xScale = d3.scaleLinear()
        .domain(xExtent)
        .range([0, innerWidth]);

      const yExtent = d3.extent(mode.loss);
      const yPadding = (yExtent[1] - yExtent[0]) * 0.1 || 0.1;
      const yScale = d3.scaleLinear()
        .domain([yExtent[0] - yPadding, yExtent[1] + yPadding])
        .range([innerHeight, 0]);

      // Grid
      g.append('g')
        .attr('class', 'grid')
        .call(d3.axisLeft(yScale).tickSize(-innerWidth).tickFormat(''));

      // Axes
      g.append('g')
        .attr('class', 'axis')
        .attr('transform', 'translate(0,' + innerHeight + ')')
        .call(d3.axisBottom(xScale).ticks(8));

      g.append('g')
        .attr('class', 'axis')
        .call(d3.axisLeft(yScale).ticks(4).tickFormat(d3.format('.3f')));

      // Axis label
      g.append('text')
        .attr('class', 'axis-label')
        .attr('transform', 'rotate(-90)')
        .attr('x', -innerHeight / 2)
        .attr('y', -45)
        .attr('text-anchor', 'middle')
        .text('loss');

      // Line
      const line = d3.line()
        .x(function(d, i) { return xScale(mode.k[i]); })
        .y(function(d) { return yScale(d); });

      g.append('path')
        .datum(mode.loss)
        .attr('class', 'line line-loss')
        .attr('d', line);

      // Hover line
      g.append('line')
        .attr('class', 'hover-line')
        .attr('id', figId + '-hover-loss')
        .attr('y1', 0)
        .attr('y2', innerHeight)
        .style('display', 'none');

      // Hover overlay
      g.append('rect')
        .attr('class', 'hover-overlay')
        .attr('id', figId + '-overlay-loss')
        .attr('width', innerWidth)
        .attr('height', innerHeight);

      svg.node().__xScale = xScale;
      svg.node().__innerHeight = innerHeight;
    }

    function renderPlotGrad(mode, N) {
      const svg = d3.select('#' + figId + '-plot-grad');
      svg.selectAll('*').remove();

      svg.attr('viewBox', '0 0 ' + width + ' ' + height)
         .attr('preserveAspectRatio', 'xMidYMid meet');

      const g = svg.append('g')
        .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

      const innerWidth = width - margin.left - margin.right;
      const innerHeight = height - margin.top - margin.bottom;

      const xExtent = d3.extent(mode.k);
      const xScale = d3.scaleLinear()
        .domain(xExtent)
        .range([0, innerWidth]);

      let allGrads = mode.grad.slice();
      if (mode.grad_fd) {
        allGrads = allGrads.concat(mode.grad_fd);
      }
      const yExtent = d3.extent(allGrads);
      const yPadding = (yExtent[1] - yExtent[0]) * 0.1 || 0.1;
      const yScale = d3.scaleLinear()
        .domain([yExtent[0] - yPadding, yExtent[1] + yPadding])
        .range([innerHeight, 0]);

      // Grid
      g.append('g')
        .attr('class', 'grid')
        .call(d3.axisLeft(yScale).tickSize(-innerWidth).tickFormat(''));

      // Axes
      g.append('g')
        .attr('class', 'axis')
        .attr('transform', 'translate(0,' + innerHeight + ')')
        .call(d3.axisBottom(xScale).ticks(8));

      g.append('g')
        .attr('class', 'axis')
        .call(d3.axisLeft(yScale).ticks(4).tickFormat(d3.format('.3f')));

      // Axis labels
      g.append('text')
        .attr('class', 'axis-label')
        .attr('x', innerWidth / 2)
        .attr('y', innerHeight + 25)
        .attr('text-anchor', 'middle')
        .text('k');

      g.append('text')
        .attr('class', 'axis-label')
        .attr('transform', 'rotate(-90)')
        .attr('x', -innerHeight / 2)
        .attr('y', -45)
        .attr('text-anchor', 'middle')
        .text('gradient');

      // Line generator
      const line = d3.line()
        .x(function(d, i) { return xScale(mode.k[i]); })
        .y(function(d) { return yScale(d); });

      // Grad line
      g.append('path')
        .datum(mode.grad)
        .attr('class', 'line line-grad')
        .attr('d', line);

      // Finite diff line
      if (mode.grad_fd) {
        g.append('path')
          .datum(mode.grad_fd)
          .attr('class', 'line line-grad-fd')
          .attr('d', line);
      }

      // Hover line
      g.append('line')
        .attr('class', 'hover-line')
        .attr('id', figId + '-hover-grad')
        .attr('y1', 0)
        .attr('y2', innerHeight)
        .style('display', 'none');

      // Hover overlay
      g.append('rect')
        .attr('class', 'hover-overlay')
        .attr('id', figId + '-overlay-grad')
        .attr('width', innerWidth)
        .attr('height', innerHeight);

      svg.node().__xScale = xScale;
      svg.node().__innerHeight = innerHeight;
    }

    function setupHover(mode, N) {
      const tooltip = document.getElementById(figId + '-tooltip');
      const svgIt = document.getElementById(figId + '-plot-it');
      const svgLoss = document.getElementById(figId + '-plot-loss');
      const svgGrad = document.getElementById(figId + '-plot-grad');

      const overlays = [
        document.getElementById(figId + '-overlay-it'),
        document.getElementById(figId + '-overlay-loss'),
        document.getElementById(figId + '-overlay-grad')
      ];

      const hoverLines = [
        document.getElementById(figId + '-hover-it'),
        document.getElementById(figId + '-hover-loss'),
        document.getElementById(figId + '-hover-grad')
      ];

      function handleMove(e) {
        const svg = e.target.closest('svg');
        if (!svg || !svg.__xScale) return;

        const rect = svg.getBoundingClientRect();
        const x = e.clientX - rect.left - margin.left * (rect.width / width);
        const xScaled = x / (rect.width / width);

        const kVal = svg.__xScale.invert(xScaled);

        // Find closest index
        let closestIdx = 0;
        let closestDist = Math.abs(mode.k[0] - kVal);
        for (let i = 1; i < N; i++) {
          const dist = Math.abs(mode.k[i] - kVal);
          if (dist < closestDist) {
            closestDist = dist;
            closestIdx = i;
          }
        }

        const xPos = svg.__xScale(mode.k[closestIdx]);

        // Show hover lines on all plots
        hoverLines.forEach(function(line) {
          line.setAttribute('x1', xPos);
          line.setAttribute('x2', xPos);
          line.style.display = 'block';
        });

        // Update tooltip
        let html = '<div class="figure-tooltip-row"><span class="figure-tooltip-label">k:</span><span class="figure-tooltip-value">' + utils.formatNumber(mode.k[closestIdx], 3) + '</span></div>';
        html += '<div class="figure-tooltip-row"><span class="figure-tooltip-label">it:</span><span class="figure-tooltip-value">' + mode.it[closestIdx] + '</span></div>';
        html += '<div class="figure-tooltip-row"><span class="figure-tooltip-label">loss:</span><span class="figure-tooltip-value">' + utils.formatNumber(mode.loss[closestIdx], 4) + '</span></div>';
        html += '<div class="figure-tooltip-row"><span class="figure-tooltip-label">grad:</span><span class="figure-tooltip-value">' + utils.formatNumber(mode.grad[closestIdx], 4) + '</span></div>';

        if (mode.grad_fd) {
          html += '<div class="figure-tooltip-row"><span class="figure-tooltip-label">grad_fd:</span><span class="figure-tooltip-value">' + utils.formatNumber(mode.grad_fd[closestIdx], 4) + '</span></div>';
        }

        tooltip.innerHTML = html;
        tooltip.classList.add('visible');

        // Position tooltip
        const containerRect = container.querySelector('.figure-plot-container').getBoundingClientRect();
        let tooltipX = e.clientX - containerRect.left + 10;
        let tooltipY = e.clientY - containerRect.top - 10;

        const tooltipRect = tooltip.getBoundingClientRect();
        if (tooltipX + tooltipRect.width > containerRect.width) {
          tooltipX = e.clientX - containerRect.left - tooltipRect.width - 10;
        }
        if (tooltipY < 0) tooltipY = 10;

        tooltip.style.left = tooltipX + 'px';
        tooltip.style.top = tooltipY + 'px';
      }

      function hideHover() {
        hoverLines.forEach(function(line) {
          line.style.display = 'none';
        });
        tooltip.classList.remove('visible');
      }

      overlays.forEach(function(overlay) {
        overlay.addEventListener('mousemove', handleMove);
        overlay.addEventListener('mouseleave', hideHover);
      });
    }
  }

  // Figure: Reduction Order Visualizer
  function initReductionOrder(config) {
    const figId = config.figId;
    const container = document.getElementById(figId);
    if (!container) return;

    const opts = config.options || {};
    let addends = opts.defaultAddends || [0.00035, 0.00028, 0.00022, 0.00018, 0.00012, 0.00008, 0.00005, 0.00003];
    let originalAddends = addends.slice();

    // Initialize controls
    const dtypeSelect = document.getElementById(figId + '-dtype');
    const policySelect = document.getElementById(figId + '-policy');
    const tolInput = document.getElementById(figId + '-tol');
    const addendsTextarea = document.getElementById(figId + '-addends');
    const largBtn = document.getElementById(figId + '-large-first');
    const smallBtn = document.getElementById(figId + '-small-first');
    const shuffleBtn = document.getElementById(figId + '-shuffle');
    const resetBtn = document.getElementById(figId + '-reset');

    // Set initial values
    addendsTextarea.value = addends.map(function(a) { return a.toFixed(6); }).join(', ');

    function parseAddends() {
      const text = addendsTextarea.value;
      const parts = text.split(/[,\s]+/).filter(function(s) { return s.length > 0; });
      const nums = parts.map(parseFloat).filter(function(n) { return !isNaN(n); });
      return nums;
    }

    function compute() {
      addends = parseAddends();
      if (addends.length === 0) return;

      const dtype = dtypeSelect.value;
      const policy = policySelect.value;
      const tol = parseFloat(tolInput.value);

      // Compute running sum with precision simulation
      const steps = [];
      let sum = 0;

      addends.forEach(function(addend, i) {
        if (policy === 'fragile') {
          // Round after each addition
          sum = precision.round(sum + addend, dtype);
        } else {
          // Accumulate in FP32, round at end
          sum = precision.toFP32(sum + addend);
        }
        steps.push({
          i: i,
          addend: addend,
          sum: sum
        });
      });

      // Final sum
      const finalSum = sum;
      const decision = finalSum < tol;

      // Update table
      const tbody = document.getElementById(figId + '-table-body');
      tbody.innerHTML = '';
      steps.forEach(function(step) {
        const tr = document.createElement('tr');
        tr.innerHTML = '<td>' + step.i + '</td><td>' + step.addend.toExponential(4) + '</td><td>' + step.sum.toExponential(6) + '</td>';
        tbody.appendChild(tr);
      });

      // Update results
      document.getElementById(figId + '-final-sum').textContent = finalSum.toExponential(6);
      document.getElementById(figId + '-tol-display').textContent = tol.toExponential(3);

      const decisionEl = document.getElementById(figId + '-decision');
      if (decision) {
        decisionEl.className = 'reduction-decision stop';
        decisionEl.textContent = 'S < tol: STOP';
      } else {
        decisionEl.className = 'reduction-decision continue';
        decisionEl.textContent = 'S >= tol: CONTINUE';
      }

      // Update margin bar
      const maxDisplay = tol * 2;
      const barWidth = Math.min(finalSum / maxDisplay * 100, 100);
      const fill = document.getElementById(figId + '-margin-fill');
      fill.style.width = barWidth + '%';
      fill.className = 'reduction-margin-fill ' + (decision ? 'under' : 'over');

      const marker = document.getElementById(figId + '-tol-marker');
      marker.style.left = (tol / maxDisplay * 100) + '%';
    }

    // Event handlers
    dtypeSelect.addEventListener('change', compute);
    policySelect.addEventListener('change', compute);
    tolInput.addEventListener('change', compute);
    addendsTextarea.addEventListener('input', compute);

    largBtn.addEventListener('click', function() {
      addends = parseAddends();
      addends.sort(function(a, b) { return b - a; });
      addendsTextarea.value = addends.map(function(a) { return a.toFixed(6); }).join(', ');
      compute();
    });

    smallBtn.addEventListener('click', function() {
      addends = parseAddends();
      addends.sort(function(a, b) { return a - b; });
      addendsTextarea.value = addends.map(function(a) { return a.toFixed(6); }).join(', ');
      compute();
    });

    shuffleBtn.addEventListener('click', function() {
      addends = parseAddends();
      // Fisher-Yates shuffle
      for (let i = addends.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        const temp = addends[i];
        addends[i] = addends[j];
        addends[j] = temp;
      }
      addendsTextarea.value = addends.map(function(a) { return a.toFixed(6); }).join(', ');
      compute();
    });

    resetBtn.addEventListener('click', function() {
      addends = originalAddends.slice();
      addendsTextarea.value = addends.map(function(a) { return a.toFixed(6); }).join(', ');
      dtypeSelect.value = 'fp32';
      policySelect.value = 'fragile';
      tolInput.value = '0.001';
      compute();
    });

    // Initial compute
    compute();
  }

  // Initialize all figures on DOMContentLoaded
  function initAllFigures() {
    if (!window.__FIGS__) return;

    window.__FIGS__.forEach(function(config) {
      if (initializedFigs.has(config.figId)) return;
      initializedFigs.add(config.figId);

      switch (config.type) {
        case 'trajectory_fork':
          initTrajectoryFork(config);
          break;
        case 'it_discontinuity':
          initItDiscontinuity(config);
          break;
        case 'reduction_order':
          initReductionOrder(config);
          break;
        default:
          console.warn('Unknown figure type:', config.type);
      }
    });
  }

  // Initialize on DOMContentLoaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAllFigures);
  } else {
    initAllFigures();
  }
})();
