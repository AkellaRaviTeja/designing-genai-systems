-- Keep the author's line breaks inside worked-arithmetic blocks (::: {.numbers}).
function Div(el)
  if el.classes:includes("numbers") then
    return el:walk({ SoftBreak = function() return pandoc.LineBreak() end })
  end
end
