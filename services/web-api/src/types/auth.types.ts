export interface AuthContext {
  user: {
    id: string;
    email: string;
  };
  tenant: {
    id: string;
    slug: string;
  };
  language: string;
  features: string[];
  permissions: string[];
}

export interface JWTPayload {
  userId: string;
  email: string;
  tenantId: string;
  tenantSlug: string;
  language: string;
  features?: string[];
  permissions?: string[];
  iat?: number;
  exp?: number;
}

export class UnauthorizedError extends Error {
  constructor(message: string = 'Unauthorized') {
    super(message);
    this.name = 'UnauthorizedError';
  }
}

export class ForbiddenError extends Error {
  constructor(message: string = 'Forbidden') {
    super(message);
    this.name = 'ForbiddenError';
  }
}

export class FeatureNotEnabledError extends Error {
  constructor(feature: string, availableFeatures: string[]) {
    super(`The '${feature}' feature is not enabled for your account`);
    this.name = 'FeatureNotEnabledError';
    this.feature = feature;
    this.availableFeatures = availableFeatures;
  }

  feature: string;
  availableFeatures: string[];
}