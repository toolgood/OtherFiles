using System.Globalization;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace ToolGood.SoarSky.StockFormer.DataProvider
{
    internal class Timefeatures
    {
        public static float[] time_features(DateTime date, string freq = "h")
        {
            Dictionary<string, List<string>> features_by_offsets = new Dictionary<string, List<string>> {
                { "Y",new List<string>(){  } },
                { "M",new List<string>(){ "MonthOfYear" } },
                { "W",new List<string>(){ "DayOfMonth", "WeekOfYear" } },
                { "D",new List<string>(){ "DayOfWeek", "DayOfMonth", "DayOfYear" } },
                { "B",new List<string>(){ "DayOfWeek", "DayOfMonth", "DayOfYear" } },
                { "H",new List<string>(){ "HourOfDay", "DayOfWeek", "DayOfMonth", "DayOfYear" } },
                { "T",new List<string>(){ "MinuteOfHour", "HourOfDay", "DayOfWeek", "DayOfMonth", "DayOfYear" } },
                { "S",new List<string>(){ "SecondOfMinute", "MinuteOfHour", "HourOfDay", "DayOfWeek", "DayOfMonth", "DayOfYear" } },

                { "L_D",new List<string>(){ "DayOfWeek", "DayOfMonth", "MonthOfYear", "DayOfYear", "L_DayOfMonth", "L_MonthOfYear", "L_DayOfYear" } },
                { "L_H",new List<string>(){ "HourOfDay", "DayOfWeek", "DayOfMonth", "MonthOfYear", "DayOfYear" , "L_DayOfMonth", "L_MonthOfYear", "L_DayOfYear" } },
                { "L_T",new List<string>(){ "MinuteOfHour", "HourOfDay", "DayOfWeek", "DayOfMonth", "MonthOfYear", "DayOfYear" , "L_DayOfMonth", "L_MonthOfYear", "L_DayOfYear" } },
                { "L_S",new List<string>(){ "SecondOfMinute", "MinuteOfHour", "HourOfDay", "DayOfWeek", "DayOfMonth", "MonthOfYear", "DayOfYear", "L_DayOfMonth", "L_MonthOfYear", "L_DayOfYear" } },
            };
            var f = features_by_offsets[freq.ToUpper()];
            var item = new float[f.Count];
            for (int i = 0; i < f.Count; i++) {
                item[i] = (float)time_features2(date, f[i]);
            }
            return item;
        }

        public static List<float[]> time_features(ICollection<DateTime> dates, string freq = "h")
        {
            var supported_freq_msg = @"
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    ";

            Dictionary<string, List<string>> features_by_offsets = new Dictionary<string, List<string>> {
                { "Y",new List<string>(){  } },
                { "M",new List<string>(){ "MonthOfYear" } },
                { "W",new List<string>(){ "DayOfMonth", "WeekOfYear" } },
                { "D",new List<string>(){ "DayOfWeek", "DayOfMonth", "DayOfYear" } },
                { "B",new List<string>(){ "DayOfWeek", "DayOfMonth", "DayOfYear" } },
                { "H",new List<string>(){ "HourOfDay", "DayOfWeek", "DayOfMonth", "DayOfYear" } },
                { "T",new List<string>(){ "MinuteOfHour", "HourOfDay", "DayOfWeek", "DayOfMonth", "DayOfYear" } },
                { "S",new List<string>(){ "SecondOfMinute", "MinuteOfHour", "HourOfDay", "DayOfWeek", "DayOfMonth", "DayOfYear" } },

                { "L_D",new List<string>(){ "DayOfWeek", "DayOfMonth", "MonthOfYear", "DayOfYear", "L_DayOfMonth", "L_MonthOfYear", "L_DayOfYear" } },
                { "L_H",new List<string>(){ "HourOfDay", "DayOfWeek", "DayOfMonth", "MonthOfYear", "DayOfYear" , "L_DayOfMonth", "L_MonthOfYear", "L_DayOfYear" } },
                { "L_T",new List<string>(){ "MinuteOfHour", "HourOfDay", "DayOfWeek", "DayOfMonth", "MonthOfYear", "DayOfYear" , "L_DayOfMonth", "L_MonthOfYear", "L_DayOfYear" } },
                { "L_S",new List<string>(){ "SecondOfMinute", "MinuteOfHour", "HourOfDay", "DayOfWeek", "DayOfMonth", "MonthOfYear", "DayOfYear", "L_DayOfMonth", "L_MonthOfYear", "L_DayOfYear" } },
            };
            var result = new List<float[]>();
            var f = features_by_offsets[freq.ToUpper()];
            foreach (var date in dates) {
                var item = new float[f.Count];
                for (int i = 0; i < f.Count; i++) {
                    item[i] = (float)time_features2(date, f[i]);
                }
                result.Add(item);
            }
            return result;
        }
        private static ChineseLunisolarCalendar chineseLunisolarCalendar = new ChineseLunisolarCalendar();

        private static double time_features2(DateTime index, string freq)
        {
            int year, month;
            switch (freq) {
                case "MonthOfYear": return (index.Month - 1) / 11.0 - 0.5;
                case "DayOfYear": return (index.DayOfYear - 1) / ((DateTime.IsLeapYear(index.Year) ? 366.0 : 365.0) - 1) - 0.5;
                case "DayOfMonth": return (index.Day - 1) / (DateTime.DaysInMonth(index.Year, index.Month) - 1) - 0.5;
                case "DayOfWeek": return (int)index.DayOfWeek / 6.0 - 0.5;
                case "HourOfDay": return index.Hour / 23.0 - 0.5;
                case "MinuteOfHour": return index.Minute / 59.0 - 0.5;
                case "SecondOfMinute": return index.Second / 59.0 - 0.5;

                case "L_MonthOfYear":
                    year = chineseLunisolarCalendar.GetYear(index);
                    month = chineseLunisolarCalendar.GetMonth(index);
                    var months = chineseLunisolarCalendar.GetMonthsInYear(year) - 1;
                    return (month - 1) / (months - 1) - 0.5;
                case "L_DayOfYear":
                    year = chineseLunisolarCalendar.GetYear(index);
                    var yearday = chineseLunisolarCalendar.GetDayOfYear(index);
                    var alldays = chineseLunisolarCalendar.GetDaysInYear(year);
                    return (yearday - 1) / (alldays - 1) - 0.5;
                case "L_DayOfMonth":
                    year = chineseLunisolarCalendar.GetYear(index);
                    month = chineseLunisolarCalendar.GetMonth(index);
                    var day = chineseLunisolarCalendar.GetDayOfMonth(index);
                    var days = chineseLunisolarCalendar.GetDaysInMonth(year, month);
                    return (day - 1) / (days - 1) - 0.5;
                default: break;
            }
            return 0;
        }



    }









}
