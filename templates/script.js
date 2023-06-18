// Suppose you have a pandas DataFrame named `df` with the necessary data

// Diagramme à barres horizontales avec Plotly.js
function drawBarH() {
    var counts = df['X'].value_counts();
    var data = [{
      x: counts.values,
      y: counts.index,
      type: 'bar',
      orientation: 'h'
    }];
    var layout = {
      title: 'Nombre de vêtements par X',
      xaxis: { title: 'Nombre de vêtements' },
      yaxis: { title: 'X' }
    };
    Plotly.newPlot('bar_h', data, layout);
  }
  
  // Diagramme en secteurs avec Plotly.js
  function drawPieChart() {
    var labels = df['EI'].value_counts().index;
    var values = df['EI'].value_counts().values;
    var data = [{
      labels: labels,
      values: values,
      type: 'pie'
    }];
    var layout = {
      title: 'Distribution du label EI (Environmental Impact)',
      title_font_size: 15
    };
    Plotly.newPlot('pie_chart', data, layout);
  }
  
  // Appel des fonctions pour générer les graphiques
  drawBarH();
  drawPieChart();
  