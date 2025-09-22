# Script for populating the database. You can run it as:
#
#     mix run priv/repo/seeds.exs
#
# Inside the script, you can read and write to any of your
# repositories directly:
#
#     CompanyManager.Repo.insert!(%CompanyManager.SomeSchema{})

alias CompanyManager.Companies

# Create sample companies
companies = [
  %{
    name: "TechCorp Industries",
    industry: "Technology",
    founded_year: 2015,
    employees: 2500,
    revenue_millions: Decimal.new("450.00")
  },
  %{
    name: "Green Energy Solutions",
    industry: "Renewable Energy",
    founded_year: 2018,
    employees: 850,
    revenue_millions: Decimal.new("125.50")
  },
  %{
    name: "FinTech Plus",
    industry: "Financial Technology",
    founded_year: 2020,
    employees: 320,
    revenue_millions: Decimal.new("75.25")
  },
  %{
    name: "MedCare Innovations",
    industry: "Healthcare",
    founded_year: 2012,
    employees: 1200,
    revenue_millions: Decimal.new("280.75")
  },
  %{
    name: "AutoTech Motors",
    industry: "Automotive",
    founded_year: 2019,
    employees: 450,
    revenue_millions: Decimal.new("95.00")
  }
]

Enum.each(companies, fn attrs ->
  case Companies.create_company(attrs) do
    {:ok, company} ->
      IO.puts("Created company: #{company.name}")
    {:error, changeset} ->
      IO.puts("Failed to create company: #{inspect(changeset.errors)}")
  end
end)