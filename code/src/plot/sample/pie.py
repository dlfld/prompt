# """
#    采样前
# """
#
# setTimeout(function()
# {
#     option = {
#     legend: {},
#     tooltip: {
#         trigger: 'axis',
#         showContent: false
#     },
#     dataset: {
#         source: [
#             ['product', '2012', '2013', '2014', '2015', '2016', '2017'],
#             ['NR', 28], ['PU', 14], ['VA', 47], ['BP', 6], ['LC', 8], ['NN', 4], ['AD', 7], ['CC', 2], ['VV', 1],
#             ['PN', 1], ['VE', 3]
#
#         ]
#     },
#     xAxis: {type: 'category'},
#     yAxis: {gridIndex: 0},
#     grid: {top: '55%'},
#     series: [
#
#         {
#             type: 'pie',
#             id: 'pie',
#             radius: '30%',
#             center: ['50%', '25%'],
#             emphasis: {
#                 focus: 'self'
#             },
#             label: {
#                 formatter: '{b}: {@2012}条，占比：{d}%'
#             },
#             encode: {
#                 itemName: 'product',
#                 value: '2012',
#                 tooltip: '2012'
#             }
#         }
#     ]
# };
#
# myChart.setOption(option);
# });
#
#
# """
#     采样后
#
# """
