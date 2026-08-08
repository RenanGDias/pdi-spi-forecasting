// Extrai exatamente a grade 13x13 e o período declarados no artigo.
// Cole no Google Earth Engine Code Editor e execute. A saída é um CSV pequeno.

var start = ee.Date('1998-01-01');
var endExclusive = ee.Date('2016-01-01');
var product = ee.ImageCollection('TRMM/3B42').select('precipitation');

var lats = ee.List.sequence(-21.00, -18.00, 0.25);
var lons = ee.List.sequence(-46.75, -43.75, 0.25);

var points = ee.FeatureCollection(lats.map(function(lat) {
  return lons.map(function(lon) {
    var row = ee.Number(lat).subtract(-21.00).divide(0.25).round();
    var col = ee.Number(lon).subtract(-46.75).divide(0.25).round();
    var pointId = row.multiply(13).add(col).add(1);
    return ee.Feature(ee.Geometry.Point([lon, lat]), {
      point_id: pointId,
      lat: lat,
      lon: lon
    });
  });
}).flatten());

// O ee.Date.difference(..., 'month') usa uma duração média do mês e, para
// este intervalo, retorna um valor ligeiramente menor que 216. Isso faz
// ee.List.sequence omitir dezembro de 2015. O artigo declara 18 anos
// completos (jan/1998 a dez/2015), portanto fixamos os 216 meses.
var monthCount = ee.Number(216);
var monthlyRows = ee.FeatureCollection(
  ee.List.sequence(0, monthCount.subtract(1)).map(function(offset) {
    var monthStart = start.advance(offset, 'month');
    var monthEnd = monthStart.advance(1, 'month');

    // A banda é taxa em mm/h. Cada imagem representa três horas.
    var monthlyMillimetres = product
      .filterDate(monthStart, monthEnd)
      .map(function(image) {
        return image.multiply(3.0).copyProperties(image, ['system:time_start']);
      })
      .sum()
      .rename('precipitation_mm');

    return monthlyMillimetres.reduceRegions({
      collection: points,
      reducer: ee.Reducer.first().setOutputs(['precipitation_mm']),
      scale: 27830,
      crs: 'EPSG:4326'
    }).map(function(feature) {
      return feature.set({
        date: monthStart.format('YYYY-MM-01'),
        year: monthStart.get('year'),
        month: monthStart.get('month')
      });
    });
  })
).flatten();

print('Número esperado de linhas (216 meses x 169 pontos):', monthlyRows.size());
print('Amostra:', monthlyRows.limit(5));

var outputSelectors = [
  'date', 'year', 'month', 'point_id', 'lat', 'lon', 'precipitation_mm'
];

// Para esta tabela pequena, também produz um link direto de download. Isso
// facilita a reprodução sem depender da fila de tarefas do Google Drive.
var directDownloadUrl = monthlyRows.getDownloadURL({
  format: 'CSV',
  selectors: outputSelectors,
  filename: 'trmm_3b42_monthly_1998_2015_upper_sao_francisco'
});
print('DOWNLOAD_URL:', directDownloadUrl);

Export.table.toDrive({
  collection: monthlyRows,
  description: 'trmm_3b42_monthly_1998_2015_upper_sao_francisco',
  fileNamePrefix: 'trmm_3b42_monthly_1998_2015_upper_sao_francisco',
  fileFormat: 'CSV',
  selectors: outputSelectors
});
