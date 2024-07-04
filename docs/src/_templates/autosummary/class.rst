:github_url: {{ fullname | escape  }}

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. add toctree option to make autodoc generate the pages

.. autoclass:: {{ objname }}

     {% block attributes %}
     {% if attributes %}
     .. rubric:: Attributes

     .. autosummary::
          :recursive:
          :toctree: .
     {% for item in attributes %}
          ~{{ fullname }}.{{ item }}
     {%- endfor %}
     {% endif %}
     {% endblock %}

     {%block functions %}
     {% if functions %}
     .. rubric:: Functions

     .. autosummary::
          :recursive:
          :toctree: .
     {% for item in functions %}
          ~{{ fullname }}.{{ item }}
     {%- endfor %}
     {% endif %}
     {% endblock %}

     {% block methods %}
     {% if methods %}
     .. rubric:: Methods

     .. autosummary::
          :toctree: .
          :recursive:
     {% for item in methods %}
          {%- if item != '__init__' %}
          ~{{ fullname }}.{{ item }}
          {%- endif -%}
     {%- endfor %}
     {% endif %}
     {% endblock %}