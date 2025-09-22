defmodule CompanyManager.Repo.Migrations.CreateCompanies do
  use Ecto.Migration

  def change do
    create table(:companies) do
      add :name, :string, null: false
      add :industry, :string, null: false
      add :founded_year, :integer
      add :employees, :integer
      add :revenue_millions, :decimal, precision: 10, scale: 2

      timestamps(type: :utc_datetime)
    end

    create index(:companies, [:industry])
    create index(:companies, [:revenue_millions])
  end
end