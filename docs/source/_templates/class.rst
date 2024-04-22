{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :no-inherited-members:
   :special-members: __call__, __add__, __mul__, forward, __init__

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nosignatures:
   {% for item in methods %}

      {%- if item.startswith('__init__') %}
        {%- if item not in inherited_members %}
        ~{{ name }}.{{ item }}
        {%- endif -%}
      {%- endif -%}
      {%- if not item.startswith('_') %}
        {%- if item not in inherited_members %}
        ~{{ name }}.{{ item }}
        {%- endif -%}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
        {%- if item not in inherited_members %}
        ~{{ name }}.{{ item }}
        {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}
