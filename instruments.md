---
display-title: "Instruments"
btn-class: "btn-annotation btn-pill"
---
<h1>Trials by Instrument</h1>
<p>Browse instruments below to find trials.</p>
{% for ct in site.instruments %}
<a class="btn btn-sm btn-pill btn-purple" href="{{ site.baseurl }}/{{ ct.url }}" title="{{ct.name}}">{{ ct.name }}</a>
{% endfor %}