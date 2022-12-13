---
layout: default
---
<h2>Browse by</h2>
<a class="btn btn-sm btn-pill btn-blue" href="{{ site.baseurl }}/labs.html">Labs</a>
<a class="btn btn-sm btn-pill btn-green" href="{{ site.baseurl }}/institutions.html">Institutions</a>
<a class="btn btn-sm btn-pill btn-purple" href="{{ site.baseurl }}/instruments.html">Instruments</a>
<br/>
<h2>Table of Contents</h2>
<ul>
{% for tr in site.trials %}
<li><a href="{{ site.baseurl }}/{{ tr.url }}">{{ tr.title }}</a> - {{ tr.description }}</li>
{% endfor %}
</ul>