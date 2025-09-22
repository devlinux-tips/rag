defmodule CompanyManager.Repo.Migrations.CreateProducts do
  use Ecto.Migration

  def change do
    create table(:products) do
      add :name, :string, null: false
      add :category, :string, null: false
      add :price, :decimal, precision: 10, scale: 2
      add :active, :boolean, default: true, null: false
      add :company_id, references(:companies, on_delete: :delete_all), null: false

      timestamps(type: :utc_datetime)
    end

    create index(:products, [:company_id])
    create index(:products, [:category])
    create index(:products, [:active])
  end
end