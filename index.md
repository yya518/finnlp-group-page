---
layout: default
title: "Home"
---

<div class="bigspacer"></div>

<div id="research-focus"></div>

<p class="research-overview">The FinNLP Group, led by <a href="https://yya518.github.io/">Prof. Yi Yang</a> at HKUST, develops NLP and AI methods for finance. Our research spans <a href="#financial-nlp-title">Financial LLM/Embedding</a>, <a href="#textual-factors-title">Textual Factors &amp; Return Predictability</a>, and <a href="#risk-allocation-title">Risk Forecasting &amp; RL for Asset Allocation</a>—from building financial language models to extracting signals from corporate disclosures 10-K filings, earnings calls, and macroeconomic narratives. We are excited to apply and scale our research with <a href="{{ site.baseurl }}/industry.html">industry partners</a>. </p>


<section class="research-feature" aria-labelledby="financial-nlp-title">
  <h2 id="financial-nlp-title">Financial LLM/Embedding</h2>
  <div class="research-feature-body">
    <a href="{{ site.baseurl }}/images/financial-embedding-v2.png" aria-label="View the financial embedding illustration at full size"><img src="{{ site.baseurl }}/images/financial-embedding-v2.png" alt="Schematic financial embedding space: Revenue rose 10% and Sales increased by 10% are close together, while Revenue fell 10% is farther away. Illustrative positions, not measured model results." width="1536" height="1024"></a>
    <p>We develop foundation models for understanding financial language. <a href="https://onlinelibrary.wiley.com/doi/full/10.1111/1911-3846.12832">FinBERT</a> extracts information and sentiment from financial text. <a href="https://arxiv.org/abs/2309.13064">InvestLM</a> adapts large language models to investment tasks through financial instruction tuning. <a href="https://aclanthology.org/2025.emnlp-main.179/">FinMTEB / FinE5</a> provides a benchmark and an embedding model for financial text representation and retrieval. <a href="{{ site.baseurl }}/papers.html#financial-llm-embedding">View more</a>.</p>
  </div>
</section>

<section class="research-feature" aria-labelledby="textual-factors-title">
  <h2 id="textual-factors-title">Textual Factors &amp; Return Predictability</h2>
  <div class="research-feature-body">
    <a href="{{ site.baseurl }}/images/textual-factors-v2.png" aria-label="View the textual factors research framework at full size"><img src="{{ site.baseurl }}/images/textual-factors-v2.png" alt="Financial disclosures, earnings calls, and macro narratives inform textual factors for return predictability and asset ranking." width="1536" height="1024"></a>
    <p>We extract factors from financial text and study their value for return prediction and asset ranking. <a href="https://arxiv.org/abs/2603.14313">Mind the Shift</a> measures changes in monetary policy stance, while <em>Departures from Routine Disclosure</em> measures changes in management outlook. Our ranking methods include <a href="https://scholar.google.com.hk/citations?view_op=view_citation&amp;hl=en&amp;user=Prh_dHkAAAAJ&amp;citation_for_view=Prh_dHkAAAAJ:V3AGJWp-ZtQC">LambdaRankIC</a>, which directly optimizes Rank IC, and <em>FinRankGRPO</em>, which trains LLMs for listwise financial asset ranking. <a href="{{ site.baseurl }}/papers.html#textual-factors">View more</a>.</p>
  </div>
</section>

<section class="research-feature" aria-labelledby="risk-allocation-title">
  <h2 id="risk-allocation-title">Risk Forecasting &amp; RL for Asset Allocation</h2>
  <div class="research-feature-body">
    <a href="{{ site.baseurl }}/images/risk-allocation.png" aria-label="View the risk and allocation illustration at full size"><img src="{{ site.baseurl }}/images/risk-allocation.png" alt="Schematic return and risk forecasts informing portfolio optimization across equity, bonds, and cash. Illustrative bars, not empirical results or recommended allocations." width="1536" height="1024"></a>
    <p>We use financial disclosures, earnings calls, and macroeconomic narratives to support risk management and assect allocation. <a href="https://scholar.google.com.hk/citations?view_op=view_citation&amp;hl=en&amp;user=Prh_dHkAAAAJ&amp;citation_for_view=Prh_dHkAAAAJ:l7t_Zn2s7bgC">Divide-and-Contrast</a> predicts firm market risk from text, and <a href="https://scholar.google.com.hk/citations?view_op=view_citation&amp;hl=en&amp;user=Prh_dHkAAAAJ&amp;citation_for_view=Prh_dHkAAAAJ:fQNAKQ3IYiAC">Learning from Earnings Calls</a> models earnings conference call structure. <a href="https://ssrn.com/abstract=7286866">MacroAllocAgent</a> translates macroeconomic narratives into strategic asset allocations. <a href="{{ site.baseurl }}/papers.html#risk-allocation">View more</a>.</p>
  </div>
</section>


<div class="bigspacer"></div>




<footer class="last-update"><span>Last update: September 2026</span><a href="{{ site.baseurl }}/contact.html">Contact</a></footer>
