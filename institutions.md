---
display-title: "Institutions"
btn-class: "btn-annotation btn-pill"
---
<h1>Trials by Institution</h1>
<p>Browse institutions below to find trials.</p>
{% for ct in site.institutions %}
<a class="btn btn-sm btn-pill btn-green" href="{{ site.baseurl }}/{{ ct.url }}" title="{{ct.name}}">{{ ct.name }}</a>
{% endfor %}