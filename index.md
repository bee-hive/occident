---
layout: home
---
<div>
  {%- include_cached header.html %}
<section id="intro"> 
  <main class="home-page-content" aria-label="Content">
    <div class="wrapper">   
      <div class="outer-container">
        <div class="row-main gy-5 py-5" style="display:flex; flex-wrap: wrap;">
          <div class="col-5 px-0 mx-0" style="display:flex;">
            <div class="bg-gradient py-1 main-top-left">
              <div class="container px-0">
                <h2 style="font-family:Arial; line-height:1.4"><b>Occident Database</b> is a website to display live cell imaging datasets and a portal for downloads.
                </h2>
                <a class="btn btn-sm btn-front my-1" href="/about.html">Read more</a>
                <br /><br />
              </div> 
            </div>
          </div>
          <div class="col-7 mx-auto gy-4 py-4 px-0" style="display:flex;">
            <div style="margin-left:39px">
              {% assign sample = site.trials | sample: 1 %} 
              <h3>Example page: <a href="{{ site.baseurl }}{{ sample.url}}">{{ sample.title}}</a> </h3>
            </div>
          </div> 
              <p style="color: #6c757d;text-align: left">More details to come...</p>
        </div>
      </div>
    </div>
  </main>
</section>
{%- include_cached footer.html -%}
<div class="wrapper">
  <div class="thanks-wrapper"> 
    {%- include_cached thanks.html -%}
  </div>
</div>

