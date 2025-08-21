// technically this is Google AppsScript! 
// https://docs.google.com/spreadsheets/d/13NriXejYairknIeESFsql4SMZNm05TKIKwxJyG4flbM/edit?gid=0#gid=0

function logWeather() {
  const lat = 40.7128;   // Latitude NYC
  const lon = -74.0060;  // Longitude NYC
  const locationName = "New York, US";
  const sheetName = "Weather_Updates"; 

  const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current_weather=true`;
  const resp = UrlFetchApp.fetch(url);
  const data = JSON.parse(resp.getContentText());
  
  const weather = data.current_weather;
  const temp = weather.temperature;   // °C
  const wind = weather.windspeed;     // m/s
  const condition = weather.weathercode; // numeric weather code
  const timeUtc = weather.time;       // UTC time string
  
  const date = new Date(timeUtc);
  const est = Utilities.formatDate(date, "America/New_York", "yyyy-MM-dd HH:mm:ss");

  const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(sheetName);
  sheet.appendRow([est, locationName, `${temp}°C, wind ${wind} m/s, code ${condition}`]);
}
