---
layout: home
title: Home
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
                <a class="btn btn-sm btn-front my-1" href="{{ site.baseurl }}/about.html">Read more</a>
                <br /><br />
              </div> 
            </div>
          </div>
          <div class="col-7 mx-auto gy-4 py-4 px-0" style="display:flex;">
            <div style="margin-left:39px">
              <h3>Example pages: </h3>
              <ul>
                <li><a href="{{ site.baseurl }}/trials/TFlib_reseed_movie_B2_Marson_2021-07-21.html">210721 - TFlib reseed movie_B2_1</a></li>
                <li><a href="{{ site.baseurl }}/trials/TFlib_reseed_movie_C4_Marson_2021-07-21.html">210721 - TFlib reseed movie_C4_1</a></li>
                <li><a href="{{ site.baseurl }}/trials/TFlib_reseed_movie_D6_Marson_2021-07-21.html">210721 - TFlib reseed movie_D6_1</a></li>

              </ul>
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