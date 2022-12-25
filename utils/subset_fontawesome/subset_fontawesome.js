// Import fontawesome-subset
const fontawesomeSubset = require("fontawesome-subset").fontawesomeSubset;

// Create or append a task to be ran with your configuration
fontawesomeSubset({
    "solid":["arrow-up", "bars", "envelope"],
    "brands":["facebook-square", "linkedin"],
}, "src/include/fontawesome/");