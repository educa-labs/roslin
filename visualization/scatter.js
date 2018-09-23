
function scatter( selector, path){
  
  var margin = { top: 50, right: 300, bottom: 50, left: 50 },
      outerWidth = 1890,
      outerHeight = 900,
      width = outerWidth - margin.left - margin.right,
      height = outerHeight - margin.top - margin.bottom;

  var x = d3.scale.linear()
      .range([0, width]).nice();

  var y = d3.scale.linear()
      .range([height, 0]).nice();

  var xCat = "0",
      yCat = "1",
      colorCat = "id";

  d3.csv(path, function(data) {
    data.forEach(function(d) {
      d.xAxis = +d[xCat]
      d.yAxis = +d[yCat]
      d.class = d[colorCat]
      d.file = d['file']
    });

    var xMax = d3.max(data, function(d) { return d.xAxis; }) * 1.05,
        xMin = d3.min(data, function(d) { return d.xAxis; }),
        xMin = xMin > 0 ? 0 : xMin,
        yMax = d3.max(data, function(d) { return d.yAxis; }) * 1.05,
        yMin = d3.min(data, function(d) { return d.yAxis; }),
        yMin = yMin > 0 ? 0 : yMin;
    x.domain([xMin, xMax]);
    y.domain([yMin, yMax]);

    var xAxis = d3.svg.axis()
        .scale(x)
        .orient("bottom")
        .tickSize(-height);

    var yAxis = d3.svg.axis()
        .scale(y)
        .orient("left")
        .tickSize(-width);

    var color = d3.scale.category10();

    var tip = d3.tip()
        .attr("class", "d3-tip")
        .offset([-10, 0])
        .html(function(d) {
          return d["id"] + "<br>" + `<img src=${d['file']}.jpg heigh=500 width=300> </img>`;
        });


    var zum = d3.behavior.zoom()
        .x(x)
        .y(y)
        .scaleExtent([0, 500])
        .on("zoom", zoom);

    var svg = d3.select(selector)
      .append("svg")
        .attr("width", outerWidth)
        .attr("height", outerHeight)
      .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
        .call(zum);

    svg.call(tip);

    svg.append("rect")
        .attr("width", width)
        .attr("height", height);

    svg.append("g")
        .classed("x axis", true)
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis)
      .append("text")
        .classed("label", true)
        .attr("x", width)
        .attr("y", margin.bottom - 10)
        .style("text-anchor", "end")
        .text("X Axis");

    svg.append("g")
        .classed("y axis", true)
        .call(yAxis)
      .append("text")
        .classed("label", true)
        .attr("transform", "rotate(-90)")
        .attr("y", -margin.left)
        .attr("dy", ".71em")
        .style("text-anchor", "end")
        .text("Y axis");

    var objects = svg.append("svg")
        .classed("objects", true)
        .attr("width", width)
        .attr("height", height);

    objects.append("svg:line")
        .classed("axisLine hAxisLine", true)
        .attr("x1", 0)
        .attr("y1", 0)
        .attr("x2", width)
        .attr("y2", 0)
        .attr("transform", "translate(0," + height + ")");

    objects.append("svg:line")
        .classed("axisLine vAxisLine", true)
        .attr("x1", 0)
        .attr("y1", 0)
        .attr("x2", 0)
        .attr("y2", height);

    objects.selectAll(".dot")
        .data(data)
      .enter().append("circle")
        .classed("dot", true)
        .attr("r", 5)
        .attr("transform", transform)
        .style("fill", 'red')
        .on("mouseover", tip.show)
        .on("mouseout", tip.hide);

    function zoom() {
      svg.select(".x.axis").call(xAxis);
      svg.select(".y.axis").call(yAxis);

      svg.selectAll(".dot")
          .attr("transform", transform);
    }

    function transform(d) {
      return "translate(" + x(d[xCat]) + "," + y(d[yCat]) + ")";
    }
  });
}